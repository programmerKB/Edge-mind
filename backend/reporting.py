"""CSV and SVG artifacts produced by dataset export and inference runs."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
import resource
import time
from typing import Any, Iterable, Sequence


REPORT_ROOT = Path(
    os.getenv(
        "INFERENCE_OUTPUT_DIR",
        Path(__file__).resolve().parent / "outputs",
    )
)
REPORT_TIMEZONE = timezone(
    timedelta(hours=float(os.getenv("REPORT_TIMEZONE_OFFSET_HOURS", "8")))
)
ANOMALY_TEMPERATURE_THRESHOLD = float(
    os.getenv("ANOMALY_TEMPERATURE_THRESHOLD", "35.0")
)
PROCESS_STARTED_AT = time.monotonic()

SENSOR_FIELDS = (
    "motor_id",
    "temperature",
    "humidity",
    "accel_x",
    "accel_y",
    "accel_z",
    "vibration",
    "status",
    "recorded_at",
)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _iso(value: datetime | None) -> str:
    return _aware(value).isoformat() if value else ""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rounded(value: float | None, digits: int = 6) -> float | str:
    return round(value, digits) if value is not None and _finite(value) else ""


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_sensor_dataset(records: Iterable[Any], path: Path) -> None:
    """Export sensor records in an Excel-friendly UTF-8 CSV file."""
    rows = []
    for record in records:
        row = {}
        for field in SENSOR_FIELDS:
            value = getattr(record, field, None)
            row[field] = _iso(value) if field == "recorded_at" else value
        rows.append(row)
    _write_csv(path, SENSOR_FIELDS, rows)


def export_demo_datasets(db) -> dict[str, str]:
    """Refresh the categorized DEMO-1 and DEMO-2 source CSV files."""
    from models import MotorSensorData
    from seed_data import DEMO_INFERENCE_MOTOR_ID, DEMO_TRAINING_MOTOR_ID

    destinations = {
        DEMO_TRAINING_MOTOR_ID: REPORT_ROOT
        / "datasets"
        / "training"
        / f"{DEMO_TRAINING_MOTOR_ID}.csv",
        DEMO_INFERENCE_MOTOR_ID: REPORT_ROOT
        / "datasets"
        / "inference"
        / f"{DEMO_INFERENCE_MOTOR_ID}.csv",
    }
    for motor_id, destination in destinations.items():
        records = (
            db.query(MotorSensorData)
            .filter(MotorSensorData.motor_id == motor_id)
            .order_by(MotorSensorData.recorded_at.asc())
            .all()
        )
        export_sensor_dataset(records, destination)
    return {key: str(value) for key, value in destinations.items()}


def _temperature_only_model(examples: Sequence[Any], ridge_alpha: float) -> dict:
    values = [float(example.features[0]) for example in examples]
    targets = [float(example.target_temperature) for example in examples]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance) if variance > 1e-12 else 1.0
    normalized = [(value - mean) / scale for value in values]
    intercept = sum(targets) / len(targets)
    weight = sum(
        value * (target - intercept)
        for value, target in zip(normalized, targets)
    ) / (sum(value * value for value in normalized) + ridge_alpha)
    return {
        "mean": mean,
        "scale": scale,
        "intercept": intercept,
        "weight": weight,
    }


def _predict_temperature_only(payload: dict, temperature: float) -> float:
    normalized = (temperature - payload["mean"]) / payload["scale"]
    return payload["intercept"] + payload["weight"] * normalized


def _regression_metrics(predictions: Sequence[float], actuals: Sequence[float]) -> dict:
    if not predictions:
        return {"sample_count": 0, "mae": None, "rmse": None, "max_error": None}
    errors = [prediction - actual for prediction, actual in zip(predictions, actuals)]
    absolute = [abs(error) for error in errors]
    return {
        "sample_count": len(errors),
        "mae": sum(absolute) / len(absolute),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "max_error": max(absolute),
    }


def _classification_metrics(
    predicted: Sequence[bool],
    actual: Sequence[bool],
    lead_minutes: int,
) -> dict:
    tp = sum(prediction and truth for prediction, truth in zip(predicted, actual))
    fp = sum(prediction and not truth for prediction, truth in zip(predicted, actual))
    tn = sum(not prediction and not truth for prediction, truth in zip(predicted, actual))
    fn = sum(not prediction and truth for prediction, truth in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarm_rate": false_alarm_rate,
        "lead_time_minutes": lead_minutes if tp else 0,
    }


def _svg_shell(title: str, body: str, width: int = 1000, height: int = 600) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<style>text{font-family:Arial,"Noto Sans TC",sans-serif;fill:#253047}'
        '.grid{stroke:#dce3ec;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.5}'
        '.label{font-size:13px}.title{font-size:24px;font-weight:700}</style>'
        f'<text x="50" y="40" class="title">{escape(title)}</text>{body}</svg>'
    )


def _write_placeholder_chart(path: Path, title: str, message: str) -> None:
    body = (
        '<rect x="80" y="100" width="840" height="400" rx="18" fill="#f1f5f9"/>'
        f'<text x="500" y="305" text-anchor="middle" font-size="20">{escape(message)}</text>'
    )
    path.write_text(_svg_shell(title, body), encoding="utf-8")


def _write_line_chart(
    path: Path,
    title: str,
    series: Sequence[tuple[str, Sequence[float], str]],
    x_labels: Sequence[str],
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [value for _, points, _ in series for value in points if _finite(value)]
    if not values or not x_labels:
        _write_placeholder_chart(path, title, "No completed observations yet")
        return
    left, top, width, height = 85, 80, 870, 420
    low, high = min(values), max(values)
    padding = max((high - low) * 0.1, 0.1)
    low, high = low - padding, high + padding

    def x_at(index: int) -> float:
        return left + (width * index / max(len(x_labels) - 1, 1))

    def y_at(value: float) -> float:
        return top + height - (value - low) / (high - low) * height

    parts = []
    for tick in range(6):
        y = top + height * tick / 5
        value = high - (high - low) * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" class="label">{value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" class="axis"/>')
    label_indexes = sorted({0, len(x_labels) // 2, len(x_labels) - 1})
    for index in label_indexes:
        parts.append(f'<text x="{x_at(index):.1f}" y="{top+height+28}" text-anchor="middle" class="label">{escape(x_labels[index])}</text>')
    parts.append(f'<text x="22" y="{top+height/2}" transform="rotate(-90 22 {top+height/2})" text-anchor="middle" class="label">{escape(y_label)}</text>')
    for series_index, (name, points, color) in enumerate(series):
        coordinates = " ".join(
            f"{x_at(index):.1f},{y_at(float(value)):.1f}"
            for index, value in enumerate(points)
        )
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        for index, value in enumerate(points):
            parts.append(f'<circle cx="{x_at(index):.1f}" cy="{y_at(float(value)):.1f}" r="3" fill="{color}"/>')
        legend_x = 650 + series_index * 150
        parts.append(f'<line x1="{legend_x}" y1="40" x2="{legend_x+28}" y2="40" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{legend_x+35}" y="45" class="label">{escape(name)}</text>')
    path.write_text(_svg_shell(title, "".join(parts)), encoding="utf-8")


def _write_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        _write_placeholder_chart(path, title, "No completed observations yet")
        return
    left, top, width, height = 85, 80, 870, 420
    high = max(max(values) * 1.2, 0.01)
    bar_slot = width / len(values)
    colors = ("#94a3b8", "#60a5fa", "#2563eb", "#f59e0b")
    parts = []
    for tick in range(6):
        y = top + height * tick / 5
        value = high * (5 - tick) / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" class="label">{value:.3f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_width = bar_slot * 0.58
        x = left + index * bar_slot + bar_slot * 0.21
        bar_height = value / high * height
        y = top + height - bar_height
        color = colors[index % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{y-9:.1f}" text-anchor="middle" class="label">{value:.4f}</text>')
        parts.append(f'<text x="{x+bar_width/2:.1f}" y="{top+height+28}" text-anchor="middle" class="label">{escape(label)}</text>')
    parts.append(f'<text x="22" y="{top+height/2}" transform="rotate(-90 22 {top+height/2})" text-anchor="middle" class="label">{escape(y_label)}</text>')
    path.write_text(_svg_shell(title, "".join(parts)), encoding="utf-8")


def _write_histogram(path: Path, title: str, errors: Sequence[float]) -> None:
    if not errors:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_placeholder_chart(path, title, "No completed observations yet")
        return
    low, high = min(errors), max(errors)
    if math.isclose(low, high):
        low, high = low - 0.05, high + 0.05
    bin_count = min(10, max(4, int(math.sqrt(len(errors)))))
    width = (high - low) / bin_count
    counts = [0] * bin_count
    for error in errors:
        index = min(int((error - low) / width), bin_count - 1)
        counts[index] += 1
    labels = [f"{low + (index + 0.5) * width:.2f}" for index in range(bin_count)]
    _write_bar_chart(path, title, labels, counts, "Count")


def _memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or "device"


def create_inference_report(
    *,
    training_records: Sequence[Any],
    inference_records: Sequence[Any],
    model_payload: dict,
    inference_motor_id: str,
    training_motor_id: str,
    latest_prediction: float,
    generated_at: datetime,
    model_inference_duration_ms: float,
    process_cpu_time_ms: float,
) -> dict:
    """Create one timestamped folder containing CSV, SVG, and JSON outputs."""
    from forecasting import (
        FORECAST_HORIZON_MINUTES,
        RIDGE_ALPHA,
        build_training_examples,
        predict_with_payload,
    )

    report_started = time.perf_counter()
    local_time = _aware(generated_at).astimezone(REPORT_TIMEZONE)
    run_directory = (
        REPORT_ROOT
        / "inference_runs"
        / local_time.strftime("%Y-%m-%d")
        / f'{local_time.strftime("%H-%M-%S-%f")}_{_safe_name(inference_motor_id)}'
    )
    csv_directory = run_directory / "csv"
    chart_directory = run_directory / "charts"
    metadata_directory = run_directory / "metadata"
    for directory in (csv_directory, chart_directory, metadata_directory):
        directory.mkdir(parents=True, exist_ok=False)

    training_examples = build_training_examples(training_records)
    evaluation_examples = build_training_examples(inference_records)
    temperature_model = _temperature_only_model(training_examples, RIDGE_ALPHA)

    five_feature_predictions = [
        predict_with_payload(model_payload, example.features)
        for example in evaluation_examples
    ]
    temperature_predictions = [
        _predict_temperature_only(temperature_model, float(example.features[0]))
        for example in evaluation_examples
    ]
    persistence_predictions = [
        float(example.features[0]) for example in evaluation_examples
    ]
    actuals = [float(example.target_temperature) for example in evaluation_examples]
    generated_iso = _iso(generated_at)

    prediction_rows = []
    for example, prediction, actual in zip(
        evaluation_examples,
        five_feature_predictions,
        actuals,
    ):
        error = prediction - actual
        prediction_rows.append(
            {
                "設備編號": inference_motor_id,
                "訓練資料集": training_motor_id,
                "預測產生時間": generated_iso,
                "來源資料時間": _iso(example.source_time),
                "預測目標時間": _iso(example.target_time),
                "預測溫度_C": round(prediction, 6),
                "30分鐘後實際溫度_C": round(actual, 6),
                "絕對誤差_C": round(abs(error), 6),
                "平方誤差_C2": round(error * error, 6),
                "資料狀態": "歷史回測_已取得真值",
            }
        )

    latest = inference_records[-1]
    latest_target = _aware(latest.recorded_at) + timedelta(
        minutes=FORECAST_HORIZON_MINUTES
    )
    prediction_rows.append(
        {
            "設備編號": inference_motor_id,
            "訓練資料集": training_motor_id,
            "預測產生時間": generated_iso,
            "來源資料時間": _iso(latest.recorded_at),
            "預測目標時間": _iso(latest_target),
            "預測溫度_C": round(latest_prediction, 6),
            "30分鐘後實際溫度_C": "",
            "絕對誤差_C": "",
            "平方誤差_C2": "",
            "資料狀態": "即時推論_等待真值",
        }
    )
    predictions_csv = csv_directory / "predictions.csv"
    _write_csv(predictions_csv, tuple(prediction_rows[0]), prediction_rows)

    methods = {
        "Persistence": persistence_predictions,
        "Temperature-only Ridge": temperature_predictions,
        "Five-feature Ridge": five_feature_predictions,
    }
    regression = {
        name: _regression_metrics(predictions, actuals)
        for name, predictions in methods.items()
    }
    main_metrics = regression["Five-feature Ridge"]
    metrics_rows = [
        {
            "設備編號": inference_motor_id,
            "訓練資料集": training_motor_id,
            "完成真值樣本數": main_metrics["sample_count"],
            "MAE_C": _rounded(main_metrics["mae"]),
            "RMSE_C": _rounded(main_metrics["rmse"]),
            "最大絕對誤差_C": _rounded(main_metrics["max_error"]),
            "異常溫度門檻_C": ANOMALY_TEMPERATURE_THRESHOLD,
        }
    ]
    metrics_csv = csv_directory / "metrics_summary.csv"
    _write_csv(metrics_csv, tuple(metrics_rows[0]), metrics_rows)

    persistence_mae = regression["Persistence"]["mae"]
    baseline_rows = []
    for name, metrics in regression.items():
        improvement = (
            (persistence_mae - metrics["mae"]) / persistence_mae * 100
            if persistence_mae
            else None
        )
        baseline_rows.append(
            {
                "方法": name,
                "用途": "30分鐘溫度預測",
                "樣本數": metrics["sample_count"],
                "MAE_C": _rounded(metrics["mae"]),
                "RMSE_C": _rounded(metrics["rmse"]),
                "最大絕對誤差_C": _rounded(metrics["max_error"]),
                "相對Persistence改善率_percent": _rounded(improvement),
                "備註": "",
            }
        )
    baseline_rows.append(
        {
            "方法": "Static temperature threshold",
            "用途": "傳統異常監控",
            "樣本數": len(actuals),
            "MAE_C": "",
            "RMSE_C": "",
            "最大絕對誤差_C": "",
            "相對Persistence改善率_percent": "",
            "備註": "門檻法為分類器，不以回歸誤差評估",
        }
    )
    baseline_csv = csv_directory / "baseline_comparison.csv"
    _write_csv(baseline_csv, tuple(baseline_rows[0]), baseline_rows)

    actual_anomalies = [
        actual >= ANOMALY_TEMPERATURE_THRESHOLD for actual in actuals
    ]
    anomaly_methods = {
        **{
            name: [
                value >= ANOMALY_TEMPERATURE_THRESHOLD for value in predictions
            ]
            for name, predictions in methods.items()
        },
        "Static temperature threshold": [
            example.features[0] >= ANOMALY_TEMPERATURE_THRESHOLD
            for example in evaluation_examples
        ],
    }
    anomaly_rows = []
    for name, predicted_anomalies in anomaly_methods.items():
        metrics = _classification_metrics(
            predicted_anomalies,
            actual_anomalies,
            FORECAST_HORIZON_MINUTES,
        )
        anomaly_rows.append(
            {
                "方法": name,
                "異常定義": f"30分鐘後溫度 >= {ANOMALY_TEMPERATURE_THRESHOLD} C",
                "真值來源": "目標時間實際溫度；不使用人工輸入的status欄位",
                **metrics,
            }
        )
    anomaly_csv = csv_directory / "anomaly_detection.csv"
    _write_csv(anomaly_csv, tuple(anomaly_rows[0]), anomaly_rows)

    source_labels = [
        _aware(example.source_time).astimezone(REPORT_TIMEZONE).strftime("%H:%M")
        for example in evaluation_examples
    ]
    signed_errors = [
        prediction - actual
        for prediction, actual in zip(five_feature_predictions, actuals)
    ]
    actual_chart = chart_directory / "01_actual_vs_predicted.svg"
    error_chart = chart_directory / "02_error_curve.svg"
    distribution_chart = chart_directory / "03_error_distribution.svg"
    baseline_chart = chart_directory / "04_baseline_mae.svg"
    anomaly_chart = chart_directory / "05_anomaly_f1.svg"
    metrics_chart = chart_directory / "06_error_metrics.svg"
    system_chart = chart_directory / "07_system_performance.svg"
    _write_line_chart(
        actual_chart,
        "Actual vs Predicted Temperature",
        (
            ("Actual", actuals, "#0f766e"),
            ("Predicted", five_feature_predictions, "#2563eb"),
        ),
        source_labels,
        "Temperature (C)",
    )
    _write_line_chart(
        error_chart,
        "Prediction Error by Observation",
        (("Prediction - Actual", signed_errors, "#dc2626"),),
        source_labels,
        "Error (C)",
    )
    _write_histogram(
        distribution_chart,
        "Prediction Error Distribution",
        signed_errors,
    )
    regression_chart_rows = [
        (name, metrics["mae"])
        for name, metrics in regression.items()
        if metrics["mae"] is not None
    ]
    _write_bar_chart(
        baseline_chart,
        "Forecast Baseline MAE Comparison",
        [name for name, _ in regression_chart_rows],
        [value for _, value in regression_chart_rows],
        "MAE (C)",
    )
    _write_bar_chart(
        anomaly_chart,
        "Anomaly Detection F1 Comparison",
        [row["方法"] for row in anomaly_rows],
        [row["f1_score"] for row in anomaly_rows],
        "F1 score",
    )
    _write_bar_chart(
        metrics_chart,
        "Five-feature Ridge Error Metrics",
        ("MAE", "RMSE", "Max error"),
        (
            (
                main_metrics["mae"],
                main_metrics["rmse"],
                main_metrics["max_error"],
            )
            if main_metrics["mae"] is not None
            else ()
        ),
        "Temperature error (C)",
    )

    total_feature_cells = len(inference_records) * 5
    missing_feature_cells = sum(
        not _finite(getattr(record, field, None))
        for record in inference_records
        for field in ("temperature", "humidity", "accel_x", "accel_y", "accel_z")
    )
    ordered_times = sorted(
        _aware(record.recorded_at)
        for record in inference_records
        if getattr(record, "recorded_at", None) is not None
    )
    expected_reading_count = (
        int(round((ordered_times[-1] - ordered_times[0]).total_seconds() / 300))
        + 1
        if len(ordered_times) >= 2
        else len(ordered_times)
    )
    missing_reading_count = max(expected_reading_count - len(ordered_times), 0)
    report_generation_duration_ms = (time.perf_counter() - report_started) * 1000
    system_rows = [
        {
            "模型訓練時間_ms": _rounded(model_payload.get("training_duration_ms")),
            "單次模型推論時間_ms": round(model_inference_duration_ms, 6),
            "報表產生時間_ms": round(report_generation_duration_ms, 6),
            "API處理時間_ms": "",
            "Agent工具呼叫正確率": "尚未建立人工標註集",
            "Agent回答與資料一致率": "尚未建立人工標註集",
            "程序CPU時間_ms": round(process_cpu_time_ms, 6),
            "程序最大RAM_MB": round(_memory_mb(), 3),
            "感測特徵遺失率_percent": round(
                missing_feature_cells / total_feature_cells * 100
                if total_feature_cells
                else 0.0,
                6,
            ),
            "感測資料筆數遺失率_percent": round(
                missing_reading_count / expected_reading_count * 100
                if expected_reading_count
                else 0.0,
                6,
            ),
            "系統連續運作時間_seconds": round(
                time.monotonic() - PROCESS_STARTED_AT,
                3,
            ),
        }
    ]
    system_csv = csv_directory / "system_performance.csv"
    _write_csv(system_csv, tuple(system_rows[0]), system_rows)
    _write_bar_chart(
        system_chart,
        "System Performance Duration",
        ("Training", "Inference", "Report", "API"),
        (
            float(model_payload.get("training_duration_ms") or 0),
            model_inference_duration_ms,
            report_generation_duration_ms,
            0.0,
        ),
        "Duration (ms)",
    )

    summary_path = metadata_directory / "run_summary.json"
    summary = {
        "run_directory": str(run_directory),
        "generated_at": generated_iso,
        "local_generated_at": local_time.isoformat(),
        "training_motor_id": training_motor_id,
        "inference_motor_id": inference_motor_id,
        "latest_prediction_c": round(latest_prediction, 6),
        "latest_actual_status": "pending",
        "completed_evaluation_samples": len(actuals),
        "metrics": main_metrics,
        "anomaly_temperature_threshold_c": ANOMALY_TEMPERATURE_THRESHOLD,
        "files": {
            "predictions_csv": str(predictions_csv),
            "metrics_csv": str(metrics_csv),
            "baseline_csv": str(baseline_csv),
            "anomaly_csv": str(anomaly_csv),
            "system_csv": str(system_csv),
            "charts": [
                str(actual_chart),
                str(error_chart),
                str(distribution_chart),
                str(baseline_chart),
                str(anomaly_chart),
                str(metrics_chart),
                str(system_chart),
            ],
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (REPORT_ROOT / "latest_run.txt").write_text(
        str(run_directory),
        encoding="utf-8",
    )
    return summary


def update_api_duration(system_csv_path: str, duration_ms: float) -> None:
    """Fill in handler timing after report creation without changing its schema."""
    path = Path(system_csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if rows:
        rows[0]["API處理時間_ms"] = round(duration_ms, 6)
        _write_csv(path, fieldnames, rows)
        _write_bar_chart(
            path.parent.parent / "charts" / "07_system_performance.svg",
            "System Performance Duration",
            ("Training", "Inference", "Report", "API"),
            (
                float(rows[0].get("模型訓練時間_ms") or 0),
                float(rows[0].get("單次模型推論時間_ms") or 0),
                float(rows[0].get("報表產生時間_ms") or 0),
                duration_ms,
            ),
            "Duration (ms)",
        )
    summary_path = path.parent.parent / "metadata" / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["api_handler_duration_ms"] = round(duration_ms, 6)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
