"""Pure comparison rules for direct Ridge and history-aware Ridge models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import time
from typing import Any, Iterable, Sequence

from edgemind.domain.evaluation import regression_metrics
from edgemind.domain.forecasting import FEATURE_NAMES, ForecastError


DIRECT_MODEL = "ridge_direct"
HISTORY_MODEL = "ridge_history"
MODEL_LABELS = {
    DIRECT_MODEL: "Direct Ridge",
    HISTORY_MODEL: "Ridge + History",
}
SAMPLING_INTERVAL_MINUTES = 5
HISTORY_MINUTES = 60
HISTORY_STEPS = HISTORY_MINUTES // SAMPLING_INTERVAL_MINUTES
FORECAST_HORIZON_MINUTES = 30
GAP_MINUTES = FORECAST_HORIZON_MINUTES
GAP_STEPS = GAP_MINUTES // SAMPLING_INTERVAL_MINUTES
ALPHA_CANDIDATES = (0.01, 0.1, 1.0)
MIN_EXPERIMENT_SAMPLES = 60

HISTORY_STATISTICS = (
    "current",
    "mean",
    "stddev",
    "min",
    "max",
    "delta",
    "slope_per_minute",
)
HISTORY_FEATURE_NAMES = tuple(
    f"{feature}_{statistic}"
    for feature in FEATURE_NAMES
    for statistic in HISTORY_STATISTICS
)


@dataclass(frozen=True, slots=True)
class RidgeExperimentExample:
    """One leakage-safe sample shared by both Ridge variants."""

    origin_time: datetime
    target_time: datetime
    current_temperature: float
    direct_features: tuple[float, ...]
    history_features: tuple[float, ...]
    target_temperature: float


@dataclass(frozen=True, slots=True)
class RidgeParameters:
    """Serializable values needed to apply a standardized Ridge model."""

    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: float
    weights: tuple[float, ...]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _finite_features(record: Any) -> tuple[float, ...] | None:
    values = tuple(getattr(record, name, None) for name in FEATURE_NAMES)
    if any(value is None for value in values):
        return None
    numeric = tuple(float(value) for value in values)
    return numeric if all(math.isfinite(value) for value in numeric) else None


def _history_vector(rows: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    """Summarize a 60-minute window without flattening all raw readings."""
    feature_values: list[float] = []
    x_values = [index * SAMPLING_INTERVAL_MINUTES for index in range(len(rows))]
    x_mean = sum(x_values) / len(x_values)
    x_variance = sum((value - x_mean) ** 2 for value in x_values)
    for feature_index in range(len(FEATURE_NAMES)):
        values = [row[feature_index] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        slope = (
            sum(
                (x_value - x_mean) * (value - mean)
                for x_value, value in zip(x_values, values)
            )
            / x_variance
            if x_variance > 0
            else 0.0
        )
        feature_values.extend(
            (
                values[-1],
                mean,
                math.sqrt(variance),
                min(values),
                max(values),
                values[-1] - values[0],
                slope,
            )
        )
    return tuple(feature_values)


def _unique_complete_readings(
    records: Iterable[Any],
) -> dict[datetime, tuple[float, ...]]:
    """Index complete timestamps and discard ambiguous duplicate timestamps."""
    grouped: dict[datetime, list[tuple[float, ...]]] = {}
    for record in records:
        recorded_at = getattr(record, "recorded_at", None)
        features = _finite_features(record)
        if recorded_at is not None and features is not None:
            grouped.setdefault(_as_utc(recorded_at), []).append(features)
    return {
        recorded_at: values[0]
        for recorded_at, values in grouped.items()
        if len(values) == 1
    }


def build_experiment_examples(
    records: Iterable[Any],
) -> list[RidgeExperimentExample]:
    """Build exact-grid samples used identically by both model variants."""
    readings = _unique_complete_readings(records)
    step = timedelta(minutes=SAMPLING_INTERVAL_MINUTES)
    horizon = timedelta(minutes=FORECAST_HORIZON_MINUTES)
    examples: list[RidgeExperimentExample] = []
    for origin_time in sorted(readings):
        history_times = [
            origin_time - step * offset
            for offset in range(HISTORY_STEPS - 1, -1, -1)
        ]
        target_time = origin_time + horizon
        if target_time not in readings or any(
            recorded_at not in readings for recorded_at in history_times
        ):
            continue
        history_rows = [readings[recorded_at] for recorded_at in history_times]
        current = history_rows[-1]
        examples.append(
            RidgeExperimentExample(
                origin_time=origin_time,
                target_time=target_time,
                current_temperature=current[0],
                direct_features=current,
                history_features=_history_vector(history_rows),
                target_temperature=readings[target_time][0],
            )
        )
    return examples


def latest_history_features(
    records: Iterable[Any],
) -> tuple[datetime, tuple[float, ...], tuple[float, ...]]:
    """Return the newest complete exact-grid window for live inference."""
    readings = _unique_complete_readings(records)
    step = timedelta(minutes=SAMPLING_INTERVAL_MINUTES)
    for origin_time in sorted(readings, reverse=True):
        history_times = [
            origin_time - step * offset
            for offset in range(HISTORY_STEPS - 1, -1, -1)
        ]
        if all(recorded_at in readings for recorded_at in history_times):
            history_rows = [readings[recorded_at] for recorded_at in history_times]
            return origin_time, history_rows[-1], _history_vector(history_rows)
    raise ForecastError(
        f"找不到連續 {HISTORY_MINUTES} 分鐘、每 {SAMPLING_INTERVAL_MINUTES} 分鐘一筆的完整資料"
    )


def chronological_split(
    examples: Sequence[RidgeExperimentExample],
) -> dict[str, list[RidgeExperimentExample]]:
    """Create a 60/20/20 chronological split with 30-minute purge gaps."""
    ordered = sorted(examples, key=lambda item: _as_utc(item.origin_time))
    if len(ordered) < MIN_EXPERIMENT_SAMPLES:
        raise ForecastError(
            f"可比較樣本只有 {len(ordered)} 筆，至少需要 {MIN_EXPERIMENT_SAMPLES} 筆"
        )
    train_stop = int(len(ordered) * 0.60)
    validation_stop = int(len(ordered) * 0.80)
    training = ordered[:train_stop]
    validation = ordered[train_stop + GAP_STEPS : validation_stop]
    testing = ordered[validation_stop + GAP_STEPS :]
    if min(len(training), len(validation), len(testing)) < 3:
        raise ForecastError("時間切分後資料不足，無法建立可靠的訓練、驗證與測試集")
    return {
        "training": training,
        "validation": validation,
        "testing": testing,
    }


def _solve_linear_system(
    matrix: list[list[float]],
    values: list[float],
) -> list[float]:
    size = len(values)
    augmented = [matrix[row][:] + [values[row]] for row in range(size)]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ForecastError("模型矩陣無法求解，請增加較多樣化的訓練資料")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
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


def fit_ridge(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    alpha: float,
) -> RidgeParameters:
    """Fit standardized Ridge parameters with an unregularized intercept."""
    if not rows or len(rows) != len(targets):
        raise ForecastError("Ridge 訓練特徵與目標筆數不一致")
    feature_count = len(rows[0])
    if any(len(row) != feature_count for row in rows):
        raise ForecastError("Ridge 訓練特徵寬度不一致")
    row_count = len(rows)
    means = tuple(
        sum(float(row[index]) for row in rows) / row_count
        for index in range(feature_count)
    )
    scales_list: list[float] = []
    for index in range(feature_count):
        variance = sum(
            (float(row[index]) - means[index]) ** 2 for row in rows
        ) / row_count
        scales_list.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    scales = tuple(scales_list)
    normalized_rows = [
        [1.0]
        + [
            (float(row[index]) - means[index]) / scales[index]
            for index in range(feature_count)
        ]
        for row in rows
    ]
    width = feature_count + 1
    gram = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row, target in zip(normalized_rows, targets):
        numeric_target = float(target)
        for left in range(width):
            rhs[left] += row[left] * numeric_target
            for right in range(left, width):
                gram[left][right] += row[left] * row[right]
    for left in range(width):
        for right in range(left):
            gram[left][right] = gram[right][left]
    for index in range(1, width):
        gram[index][index] += float(alpha)
    coefficients = _solve_linear_system(gram, rhs)
    return RidgeParameters(
        means=means,
        scales=scales,
        intercept=coefficients[0],
        weights=tuple(coefficients[1:]),
    )


def predict_ridge(parameters: RidgeParameters, features: Sequence[float]) -> float:
    if len(features) != len(parameters.weights):
        raise ForecastError("Ridge 推論特徵數量不正確")
    return float(
        parameters.intercept
        + sum(
            weight * ((float(value) - mean) / scale)
            for weight, value, mean, scale in zip(
                parameters.weights,
                features,
                parameters.means,
                parameters.scales,
            )
        )
    )


def _feature_row(example: RidgeExperimentExample, model_name: str):
    return (
        example.direct_features
        if model_name == DIRECT_MODEL
        else example.history_features
    )


def _predicted_temperature(
    parameters: RidgeParameters,
    example: RidgeExperimentExample,
    model_name: str,
) -> float:
    predicted_delta = predict_ridge(parameters, _feature_row(example, model_name))
    return example.current_temperature + predicted_delta


def _targets(examples: Sequence[RidgeExperimentExample]) -> list[float]:
    return [
        example.target_temperature - example.current_temperature
        for example in examples
    ]


def _evaluate(
    parameters: RidgeParameters,
    examples: Sequence[RidgeExperimentExample],
    model_name: str,
) -> tuple[dict, list[float]]:
    predictions = [
        _predicted_temperature(parameters, example, model_name)
        for example in examples
    ]
    return (
        regression_metrics(
            predictions,
            [example.target_temperature for example in examples],
        ),
        predictions,
    )


def _select_and_evaluate(
    split: dict[str, list[RidgeExperimentExample]],
    model_name: str,
) -> tuple[dict, RidgeParameters, list[dict]]:
    training = split["training"]
    validation = split["validation"]
    testing = split["testing"]
    candidate_results = []
    for alpha in ALPHA_CANDIDATES:
        parameters = fit_ridge(
            [_feature_row(example, model_name) for example in training],
            _targets(training),
            alpha,
        )
        metrics, _ = _evaluate(parameters, validation, model_name)
        candidate_results.append({"alpha": alpha, "metrics": metrics})
    selected = min(candidate_results, key=lambda item: item["metrics"]["mae"])
    development = [*training, *validation]
    final_parameters = fit_ridge(
        [_feature_row(example, model_name) for example in development],
        _targets(development),
        selected["alpha"],
    )
    test_metrics, predictions = _evaluate(final_parameters, testing, model_name)
    feature_names = (
        FEATURE_NAMES if model_name == DIRECT_MODEL else HISTORY_FEATURE_NAMES
    )
    result = {
        "model_name": model_name,
        "label": MODEL_LABELS[model_name],
        "selected_alpha": selected["alpha"],
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "validation_metrics": selected["metrics"],
        "test_metrics": test_metrics,
        "alpha_search": candidate_results,
    }
    prediction_rows = [
        {
            "model_name": model_name,
            "origin_time": example.origin_time.isoformat(),
            "target_time": example.target_time.isoformat(),
            "actual_temperature": example.target_temperature,
            "predicted_temperature": prediction,
            "error": prediction - example.target_temperature,
        }
        for example, prediction in zip(testing, predictions)
    ]
    return result, final_parameters, prediction_rows


def run_ridge_experiment(
    training_records: Iterable[Any],
    *,
    external_records: Iterable[Any] | None = None,
) -> dict:
    """Run the locked comparison and optional zero-shot external evaluation."""
    started = time.perf_counter()
    examples = build_experiment_examples(training_records)
    split = chronological_split(examples)
    model_results: list[dict] = []
    fitted_models: dict[str, RidgeParameters] = {}
    prediction_rows: list[dict] = []
    for model_name in (DIRECT_MODEL, HISTORY_MODEL):
        result, parameters, predictions = _select_and_evaluate(split, model_name)
        model_results.append(result)
        fitted_models[model_name] = parameters
        prediction_rows.extend(predictions)

    external_examples = (
        build_experiment_examples(external_records)
        if external_records is not None
        else []
    )
    if external_records is not None and not external_examples:
        raise ForecastError("外部評估設備沒有可用的連續歷史樣本")
    for result in model_results:
        if external_examples:
            external_metrics, _ = _evaluate(
                fitted_models[result["model_name"]],
                external_examples,
                result["model_name"],
            )
            result["external_metrics"] = external_metrics

    direct_mae = model_results[0]["test_metrics"]["mae"]
    history_mae = model_results[1]["test_metrics"]["mae"]
    improvement = (
        (direct_mae - history_mae) / direct_mae * 100
        if direct_mae and direct_mae > 1e-12
        else None
    )
    return {
        "status": "completed",
        "configuration": experiment_configuration(),
        "dataset": {
            "usable_sample_count": len(examples),
            "training_sample_count": len(split["training"]),
            "validation_sample_count": len(split["validation"]),
            "test_sample_count": len(split["testing"]),
            "purged_sample_count": len(examples)
            - sum(len(part) for part in split.values()),
            "external_sample_count": len(external_examples),
            "first_origin_time": examples[0].origin_time.isoformat(),
            "last_target_time": examples[-1].target_time.isoformat(),
        },
        "models": model_results,
        "comparison": {
            "winner": (
                HISTORY_MODEL if history_mae < direct_mae else DIRECT_MODEL
            ),
            "test_mae_improvement_percent": improvement,
            "history_minus_direct_mae": history_mae - direct_mae,
        },
        "predictions": prediction_rows,
        "duration_ms": (time.perf_counter() - started) * 1000,
    }


def experiment_configuration() -> dict:
    """Return the fixed, auditable experiment contract for the UI."""
    return {
        "models": [
            {
                "model_name": DIRECT_MODEL,
                "label": MODEL_LABELS[DIRECT_MODEL],
                "description": "只使用預測起點當下的 5 個感測特徵。",
                "feature_count": len(FEATURE_NAMES),
            },
            {
                "model_name": HISTORY_MODEL,
                "label": MODEL_LABELS[HISTORY_MODEL],
                "description": "使用最近 60 分鐘各特徵的現值、均值、波動、極值、變化量與斜率。",
                "feature_count": len(HISTORY_FEATURE_NAMES),
            },
        ],
        "sampling_interval_minutes": SAMPLING_INTERVAL_MINUTES,
        "history_minutes": HISTORY_MINUTES,
        "history_steps": HISTORY_STEPS,
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "split": {"training": 0.6, "validation": 0.2, "testing": 0.2},
        "gap_minutes": GAP_MINUTES,
        "alpha_candidates": list(ALPHA_CANDIDATES),
        "comparison_metric": "test_mae",
    }


def train_for_live_forecast(
    training_records: Iterable[Any],
    model_name: str,
) -> tuple[RidgeParameters, dict]:
    """Select alpha chronologically, then refit one model on all history."""
    if model_name not in MODEL_LABELS:
        raise ForecastError(f"不支援的模型：{model_name}")
    examples = build_experiment_examples(training_records)
    split = chronological_split(examples)
    model_result, _, _ = _select_and_evaluate(split, model_name)
    parameters = fit_ridge(
        [_feature_row(example, model_name) for example in examples],
        _targets(examples),
        model_result["selected_alpha"],
    )
    return parameters, {
        "selected_alpha": model_result["selected_alpha"],
        "sample_count": len(examples),
        "feature_count": model_result["feature_count"],
        "validation_metrics": model_result["validation_metrics"],
        "test_metrics": model_result["test_metrics"],
    }
