"""Pure Ridge-regression calculations for 30-minute temperature forecasts.

This module intentionally has no database or filesystem imports. Plain sensor
objects can therefore be evaluated quickly in unit tests or reused on an edge
device without starting FastAPI.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time
from typing import Any, Iterable, Sequence

from evaluation import regression_metrics


FEATURE_NAMES = ("temperature", "humidity", "accel_x", "accel_y", "accel_z")
FORECAST_HORIZON_MINUTES = 30
TARGET_TOLERANCE_MINUTES = 5
MIN_TRAINING_SAMPLES = 12
RIDGE_ALPHA = 0.01


class ForecastError(ValueError):
    """Raised when sensor history cannot produce a trustworthy forecast."""


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One feature vector paired with its future temperature target."""

    source_time: datetime
    target_time: datetime
    features: tuple[float, ...]
    target_temperature: float


def _as_utc_timestamp(value: datetime) -> float:
    """Normalize naive database timestamps before chronological comparison."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def extract_sensor_features(record: Any) -> tuple[float, ...] | None:
    """Return a finite five-feature vector or ``None`` for incomplete rows."""
    values = tuple(getattr(record, name, None) for name in FEATURE_NAMES)
    if any(value is None for value in values):
        return None
    numeric = tuple(float(value) for value in values)
    return numeric if all(math.isfinite(value) for value in numeric) else None


def build_training_examples(
    records: Iterable[Any],
    horizon_minutes: int = FORECAST_HORIZON_MINUTES,
    tolerance_minutes: int = TARGET_TOLERANCE_MINUTES,
) -> list[TrainingExample]:
    """Pair complete readings with the closest target around ``t + horizon``.

    Feature extraction is performed once per row. The previous implementation
    evaluated every valid source row twice, which was unnecessary for larger
    sensor histories.
    """
    complete: list[tuple[Any, tuple[float, ...]]] = []
    for record in records:
        features = extract_sensor_features(record)
        if record.recorded_at is not None and features is not None:
            complete.append((record, features))
    complete.sort(key=lambda item: _as_utc_timestamp(item[0].recorded_at))

    timestamps = [_as_utc_timestamp(record.recorded_at) for record, _ in complete]
    horizon_seconds = horizon_minutes * 60
    tolerance_seconds = tolerance_minutes * 60
    examples: list[TrainingExample] = []

    # Binary search keeps target matching O(n log n), rather than scanning all
    # later readings for every source sample.
    for source_index, (source, features) in enumerate(complete):
        expected = timestamps[source_index] + horizon_seconds
        insertion = bisect_left(timestamps, expected, lo=source_index + 1)
        candidate_indexes = (
            index
            for index in (insertion - 1, insertion)
            if source_index < index < len(complete)
        )
        try:
            target_index = min(
                candidate_indexes,
                key=lambda index: abs(timestamps[index] - expected),
            )
        except ValueError:
            continue
        if abs(timestamps[target_index] - expected) > tolerance_seconds:
            continue
        target = complete[target_index][0]
        examples.append(
            TrainingExample(
                source_time=source.recorded_at,
                target_time=target.recorded_at,
                features=features,
                target_temperature=float(target.temperature),
            )
        )
    return examples


def _solve_linear_system(
    matrix: list[list[float]],
    values: list[float],
) -> list[float]:
    """Solve ``Ax=b`` using Gaussian elimination with partial pivoting."""
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
        augmented[column] = [value / divisor for value in augmented[column]]
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
    """Fit standardized Ridge parameters without regularizing the intercept."""
    feature_count = len(FEATURE_NAMES)
    example_count = len(examples)
    means = [
        sum(example.features[index] for example in examples) / example_count
        for index in range(feature_count)
    ]
    scales = []
    for index in range(feature_count):
        variance = sum(
            (example.features[index] - means[index]) ** 2
            for example in examples
        ) / example_count
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
        [sum(row[i] * row[j] for row in rows) for j in range(width)]
        for i in range(width)
    ]
    for index in range(1, width):
        gram[index][index] += RIDGE_ALPHA
    rhs = [
        sum(row[index] * target for row, target in zip(rows, targets))
        for index in range(width)
    ]
    coefficients = _solve_linear_system(gram, rhs)
    return {
        "means": means,
        "scales": scales,
        "intercept": coefficients[0],
        "weights": coefficients[1:],
    }


def predict_with_payload(payload: dict, features: Sequence[float]) -> float:
    """Apply one serialized Ridge model to a five-feature input vector."""
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


def train_temperature_model(examples: Sequence[TrainingExample]) -> dict:
    """Time-split, validate, then refit a compact serializable Ridge model."""
    training_started = time.perf_counter()
    if len(examples) < MIN_TRAINING_SAMPLES:
        raise ForecastError(
            f"有效的 30 分鐘訓練樣本只有 {len(examples)} 筆，"
            f"至少需要 {MIN_TRAINING_SAMPLES} 筆"
        )

    ordered = sorted(examples, key=lambda item: _as_utc_timestamp(item.source_time))
    validation_size = max(2, int(round(len(ordered) * 0.2)))
    training = ordered[:-validation_size]
    validation = ordered[-validation_size:]
    validation_model = _fit_parameters(training)
    validation_predictions = [
        predict_with_payload(validation_model, example.features)
        for example in validation
    ]
    validation_metrics = regression_metrics(
        validation_predictions,
        [example.target_temperature for example in validation],
    )

    # The time split measures future-like performance; the final model is then
    # refit on all valid history to maximize the data available in production.
    payload = _fit_parameters(ordered)
    payload.update(
        {
            "version": 2,
            "algorithm": "ridge_regression",
            "feature_names": list(FEATURE_NAMES),
            "horizon_minutes": FORECAST_HORIZON_MINUTES,
            "target_tolerance_minutes": TARGET_TOLERANCE_MINUTES,
            "sample_count": len(ordered),
            "validation_sample_count": len(validation),
            "mae": validation_metrics["mae"],
            "rmse": validation_metrics["rmse"],
            "validation_metrics": validation_metrics,
            "training_duration_ms": (
                time.perf_counter() - training_started
            )
            * 1000,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return payload
