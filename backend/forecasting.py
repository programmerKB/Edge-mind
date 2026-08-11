"""30-minute temperature forecasting from environmental sensor readings.

The model is regularized linear regression implemented with the Python
standard library. Keeping the serialized model as JSON makes training and
inference deterministic and avoids a large binary ML dependency on edge
devices.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import time
from typing import Any, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from models import MotorSensorData


FEATURE_NAMES = ("temperature", "humidity", "accel_x", "accel_y", "accel_z")
FORECAST_HORIZON_MINUTES = 30
TARGET_TOLERANCE_MINUTES = 5
MIN_TRAINING_SAMPLES = 12
RIDGE_ALPHA = 0.01


class ForecastError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingExample:
    source_time: datetime
    target_time: datetime
    features: tuple[float, ...]
    target_temperature: float


def _as_utc_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _sensor_features(record: Any) -> tuple[float, ...] | None:
    values = tuple(getattr(record, name, None) for name in FEATURE_NAMES)
    if any(value is None for value in values):
        return None
    numeric = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numeric):
        return None
    return numeric


def build_training_examples(
    records: Iterable[Any],
    horizon_minutes: int = FORECAST_HORIZON_MINUTES,
    tolerance_minutes: int = TARGET_TOLERANCE_MINUTES,
) -> list[TrainingExample]:
    """Pair each complete reading with the closest reading around t + 30 min."""
    complete = [
        record
        for record in records
        if record.recorded_at and _sensor_features(record) is not None
    ]
    complete.sort(key=lambda record: _as_utc_timestamp(record.recorded_at))
    timestamps = [_as_utc_timestamp(record.recorded_at) for record in complete]
    horizon_seconds = horizon_minutes * 60
    tolerance_seconds = tolerance_minutes * 60
    examples: list[TrainingExample] = []

    for source_index, source in enumerate(complete):
        expected = timestamps[source_index] + horizon_seconds
        insertion = bisect_left(timestamps, expected, lo=source_index + 1)
        candidate_indexes = [
            index
            for index in (insertion - 1, insertion)
            if source_index < index < len(complete)
        ]
        if not candidate_indexes:
            continue
        target_index = min(
            candidate_indexes,
            key=lambda index: abs(timestamps[index] - expected),
        )
        if abs(timestamps[target_index] - expected) > tolerance_seconds:
            continue
        target = complete[target_index]
        examples.append(
            TrainingExample(
                source_time=source.recorded_at,
                target_time=target.recorded_at,
                features=_sensor_features(source),
                target_temperature=float(target.temperature),
            )
        )
    return examples


def _solve_linear_system(
    matrix: list[list[float]],
    values: list[float],
) -> list[float]:
    """Solve Ax=b using Gaussian elimination with partial pivoting."""
    size = len(values)
    augmented = [matrix[row][:] + [values[row]] for row in range(size)]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ForecastError("模型矩陣無法求解，請增加較多樣化的訓練資料")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [
            value / divisor for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][index]
                    - factor * augmented[column][index]
                    for index in range(size + 1)
                ]
    return [augmented[row][-1] for row in range(size)]


def _fit_parameters(examples: Sequence[TrainingExample]) -> dict:
    feature_count = len(FEATURE_NAMES)
    means = [
        sum(example.features[index] for example in examples) / len(examples)
        for index in range(feature_count)
    ]
    scales = []
    for index in range(feature_count):
        variance = (
            sum(
                (example.features[index] - means[index]) ** 2
                for example in examples
            )
            / len(examples)
        )
        scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

    rows = [
        [1.0]
        + [
            (example.features[index] - means[index]) / scales[index]
            for index in range(feature_count)
        ]
        for example in examples
    ]
    targets = [example.target_temperature for example in examples]
    width = feature_count + 1
    gram = [
        [
            sum(row[i] * row[j] for row in rows)
            for j in range(width)
        ]
        for i in range(width)
    ]
    # Do not regularize the intercept.
    for index in range(1, width):
        gram[index][index] += RIDGE_ALPHA
    rhs = [
        sum(
            row[index] * target
            for row, target in zip(rows, targets)
        )
        for index in range(width)
    ]
    coefficients = _solve_linear_system(gram, rhs)
    return {
        "means": means,
        "scales": scales,
        "intercept": coefficients[0],
        "weights": coefficients[1:],
    }


def predict_with_payload(
    payload: dict,
    features: Sequence[float],
) -> float:
    if len(features) != len(FEATURE_NAMES):
        raise ForecastError("輸入特徵數量不正確")
    normalized = [
        (float(features[index]) - payload["means"][index])
        / payload["scales"][index]
        for index in range(len(FEATURE_NAMES))
    ]
    return float(
        payload["intercept"]
        + sum(
            weight * value
            for weight, value in zip(payload["weights"], normalized)
        )
    )


def train_temperature_model(
    examples: Sequence[TrainingExample],
) -> dict:
    training_started = time.perf_counter()
    if len(examples) < MIN_TRAINING_SAMPLES:
        raise ForecastError(
            f"有效的 30 分鐘訓練樣本只有 {len(examples)} 筆，"
            f"至少需要 {MIN_TRAINING_SAMPLES} 筆"
        )

    ordered = sorted(
        examples,
        key=lambda example: _as_utc_timestamp(example.source_time),
    )
    validation_size = max(2, int(round(len(ordered) * 0.2)))
    training = ordered[:-validation_size]
    validation = ordered[-validation_size:]
    validation_model = _fit_parameters(training)
    residuals = [
        predict_with_payload(validation_model, example.features)
        - example.target_temperature
        for example in validation
    ]
    mae = sum(abs(value) for value in residuals) / len(residuals)
    rmse = math.sqrt(
        sum(value * value for value in residuals) / len(residuals)
    )

    # Refit on all available history after the time-ordered evaluation.
    payload = _fit_parameters(ordered)
    payload.update(
        {
            "version": 1,
            "algorithm": "ridge_regression",
            "feature_names": list(FEATURE_NAMES),
            "horizon_minutes": FORECAST_HORIZON_MINUTES,
            "target_tolerance_minutes": TARGET_TOLERANCE_MINUTES,
            "sample_count": len(ordered),
            "validation_sample_count": len(validation),
            "mae": mae,
            "rmse": rmse,
            "training_duration_ms": (
                time.perf_counter() - training_started
            )
            * 1000,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return payload


def train_and_save_model(db, motor_id: str) -> dict:
    from models import MotorSensorData, TemperatureForecastModel

    records = (
        db.query(MotorSensorData)
        .filter(MotorSensorData.motor_id == motor_id)
        .order_by(MotorSensorData.recorded_at.asc())
        .all()
    )
    payload = train_temperature_model(build_training_examples(records))
    stored = (
        db.query(TemperatureForecastModel)
        .filter(TemperatureForecastModel.motor_id == motor_id)
        .first()
    )
    if stored is None:
        stored = TemperatureForecastModel(
            motor_id=motor_id,
            model_json="{}",
            sample_count=0,
        )
        db.add(stored)
    stored.model_json = json.dumps(payload, ensure_ascii=False)
    stored.sample_count = payload["sample_count"]
    stored.mae = payload["mae"]
    stored.rmse = payload["rmse"]
    stored.trained_at = datetime.now(timezone.utc)
    db.commit()
    return payload


def forecast_temperature(
    db,
    motor_id: str,
    auto_train: bool = True,
    training_motor_id: str | None = None,
) -> dict:
    from models import MotorSensorData, TemperatureForecastModel
    from reporting import create_inference_report

    inference_started = time.perf_counter()
    cpu_started = time.process_time()
    model_motor_id = training_motor_id or motor_id
    stored = (
        db.query(TemperatureForecastModel)
        .filter(TemperatureForecastModel.motor_id == model_motor_id)
        .first()
    )
    if stored is None:
        if not auto_train:
            raise ForecastError(f"設備 {model_motor_id} 尚未訓練預測模型")
        train_and_save_model(db, model_motor_id)
        stored = (
            db.query(TemperatureForecastModel)
            .filter(TemperatureForecastModel.motor_id == model_motor_id)
            .first()
        )

    latest = (
        db.query(MotorSensorData)
        .filter(MotorSensorData.motor_id == motor_id)
        .order_by(MotorSensorData.recorded_at.desc())
        .first()
    )
    if latest is None:
        raise ForecastError(f"資料庫中找不到設備 {motor_id} 的感測資料")
    features = _sensor_features(latest)
    if features is None:
        raise ForecastError(
            f"設備 {motor_id} 最新資料缺少溫度、濕度或三軸 XYZ 特徵"
        )

    payload = json.loads(stored.model_json)
    prediction = predict_with_payload(payload, features)
    model_inference_duration_ms = (
        time.perf_counter() - inference_started
    ) * 1000
    process_cpu_time_ms = (time.process_time() - cpu_started) * 1000
    source_time = latest.recorded_at
    target_time = source_time + timedelta(
        minutes=FORECAST_HORIZON_MINUTES
    )
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
        "model": {
            "training_motor_id": model_motor_id,
            "algorithm": payload["algorithm"],
            "sample_count": payload["sample_count"],
            "validation_mae": round(payload["mae"], 4),
            "validation_rmse": round(payload["rmse"], 4),
            "training_duration_ms": (
                round(payload["training_duration_ms"], 6)
                if payload.get("training_duration_ms") is not None
                else None
            ),
            "trained_at": payload["trained_at"],
        },
    }
    training_records = (
        db.query(MotorSensorData)
        .filter(MotorSensorData.motor_id == model_motor_id)
        .order_by(MotorSensorData.recorded_at.asc())
        .all()
    )
    inference_records = (
        db.query(MotorSensorData)
        .filter(MotorSensorData.motor_id == motor_id)
        .order_by(MotorSensorData.recorded_at.asc())
        .all()
    )
    report = create_inference_report(
        training_records=training_records,
        inference_records=inference_records,
        model_payload=payload,
        inference_motor_id=motor_id,
        training_motor_id=model_motor_id,
        latest_prediction=prediction,
        generated_at=datetime.now(timezone.utc),
        model_inference_duration_ms=model_inference_duration_ms,
        process_cpu_time_ms=process_cpu_time_ms,
    )
    evaluation_metrics = report["metrics"]
    result["evaluation"] = {
        "completed_samples": report["completed_evaluation_samples"],
        "mae": (
            round(evaluation_metrics["mae"], 6)
            if evaluation_metrics["mae"] is not None
            else None
        ),
        "rmse": (
            round(evaluation_metrics["rmse"], 6)
            if evaluation_metrics["rmse"] is not None
            else None
        ),
        "max_error": (
            round(evaluation_metrics["max_error"], 6)
            if evaluation_metrics["max_error"] is not None
            else None
        ),
    }
    result["artifacts"] = {
        "run_directory": report["run_directory"],
        **report["files"],
    }
    return result
