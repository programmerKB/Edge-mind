"""Reusable model-evaluation and descriptive-statistics helpers."""

from __future__ import annotations

import math
from statistics import fmean, median, pstdev
from typing import Iterable, Sequence


def _paired_values(
    predicted: Sequence,
    actual: Sequence,
) -> list[tuple]:
    """Validate equal lengths once before calculating paired metrics."""
    if len(predicted) != len(actual):
        raise ValueError("預測值與真值筆數必須相同")
    return list(zip(predicted, actual))


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return zero for undefined confusion-matrix ratios."""
    return numerator / denominator if denominator else 0.0


def _percentile_from_sorted(
    ordered: Sequence[float],
    probability: float,
) -> float | None:
    """Interpolate a percentile from values sorted by the caller."""
    if not 0 <= probability <= 1:
        raise ValueError("percentile probability 必須介於 0 與 1")
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Return a linearly interpolated percentile for probability 0..1."""
    ordered = sorted(
        float(value) for value in values if math.isfinite(float(value))
    )
    return _percentile_from_sorted(ordered, probability)


def descriptive_statistics(values: Iterable[float]) -> dict:
    """Summarize repeated measurements without hiding tail latency."""
    ordered = sorted(
        float(value) for value in values if math.isfinite(float(value))
    )
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev": None,
            "min": None,
            "max": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(ordered),
        "mean": fmean(ordered),
        "median": median(ordered),
        "stddev": pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        # Reuse one sort for all tail percentiles in the summary.
        "p90": _percentile_from_sorted(ordered, 0.90),
        "p95": _percentile_from_sorted(ordered, 0.95),
        "p99": _percentile_from_sorted(ordered, 0.99),
    }


def regression_metrics(
    predictions: Sequence[float],
    actuals: Sequence[float],
) -> dict:
    """Calculate common regression metrics for one evaluated model."""
    pairs = _paired_values(predictions, actuals)
    if not pairs:
        return {
            "sample_count": 0,
            "mae": None,
            "median_absolute_error": None,
            "mse": None,
            "rmse": None,
            "max_error": None,
            "r2_score": None,
            "mape_percent": None,
            "mean_error": None,
        }

    predicted_values = [float(prediction) for prediction, _ in pairs]
    actual_values = [float(actual) for _, actual in pairs]
    errors = [
        prediction - actual
        for prediction, actual in zip(predicted_values, actual_values)
    ]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    actual_mean = fmean(actual_values)
    total_variance = sum((actual - actual_mean) ** 2 for actual in actual_values)
    nonzero_percentage_errors = [
        abs(error / actual) * 100
        for error, actual in zip(errors, actual_values)
        if not math.isclose(actual, 0.0, abs_tol=1e-12)
    ]
    return {
        "sample_count": len(errors),
        "mae": fmean(absolute_errors),
        "median_absolute_error": median(absolute_errors),
        "mse": fmean(squared_errors),
        "rmse": math.sqrt(fmean(squared_errors)),
        "max_error": max(absolute_errors),
        "r2_score": (
            1 - sum(squared_errors) / total_variance
            if total_variance > 1e-12
            else None
        ),
        "mape_percent": (
            fmean(nonzero_percentage_errors)
            if nonzero_percentage_errors
            else None
        ),
        "mean_error": fmean(errors),
    }


def classification_metrics(
    predicted: Sequence[bool],
    actual: Sequence[bool],
    lead_minutes: int = 0,
) -> dict:
    """Calculate confusion-matrix metrics for binary anomaly detection."""
    pairs = _paired_values(predicted, actual)
    tp = fp = tn = fn = 0
    # One pass is clearer and avoids walking a large classification set four
    # separate times.
    for prediction, truth in pairs:
        if bool(prediction):
            if bool(truth):
                tp += 1
            else:
                fp += 1
        elif bool(truth):
            fn += 1
        else:
            tn += 1
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    negative_predictive_value = _safe_divide(tn, tn + fn)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": _safe_divide(tp + tn, len(pairs)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "f1_score": _safe_divide(2 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
        "false_alarm_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
        "positive_support": tp + fn,
        "negative_support": tn + fp,
        "lead_time_minutes": lead_minutes if tp else 0,
    }


def binary_score_metrics(scores: Sequence[float], actual: Sequence[bool]) -> dict:
    """Calculate threshold-independent ROC-AUC, PR-AUC, and average precision."""
    pairs = _paired_values(scores, actual)
    positives = sum(bool(truth) for _, truth in pairs)
    negatives = len(pairs) - positives
    if not pairs or not positives or not negatives:
        return {"roc_auc": None, "pr_auc": None, "average_precision": None}

    ordered = sorted(
        ((float(score), bool(truth)) for score, truth in pairs),
        key=lambda pair: pair[0],
        reverse=True,
    )
    roc_points = [(0.0, 0.0)]
    pr_points = [(0.0, 1.0)]
    average_precision = 0.0
    tp = fp = 0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_tp = group_fp = 0
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1]:
                group_tp += 1
            else:
                group_fp += 1
            index += 1
        tp += group_tp
        fp += group_fp
        recall = tp / positives
        precision = tp / (tp + fp)
        roc_points.append((fp / negatives, recall))
        pr_points.append((recall, precision))
        average_precision += group_tp / positives * precision

    def area(points: Sequence[tuple[float, float]]) -> float:
        return sum(
            (right_x - left_x) * (left_y + right_y) / 2
            for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
        )

    return {
        "roc_auc": area(roc_points),
        "pr_auc": area(pr_points),
        "average_precision": average_precision,
    }
