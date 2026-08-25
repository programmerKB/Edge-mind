"""Orchestrate one inference evaluation and its CSV/SVG artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
import re
import resource
import time
from typing import Any, Sequence

from evaluation import (
    binary_score_metrics,
    classification_metrics,
    regression_metrics,
)
from forecast.engine import (
    FORECAST_HORIZON_MINUTES,
    RIDGE_ALPHA,
    build_training_examples,
    predict_with_payload,
)
from reports import config
from reports.charts import write_bar_chart, write_histogram, write_line_chart
from reports.io import aware_datetime, is_finite, iso_datetime, rounded, write_csv


def train_temperature_baseline(examples: Sequence[Any], ridge_alpha: float) -> dict:
    """Fit a one-feature Ridge baseline for honest model comparison."""
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


def predict_temperature_baseline(payload: dict, temperature: float) -> float:
    """Apply the temperature-only comparison model to one reading."""
    normalized = (temperature - payload["mean"]) / payload["scale"]
    return payload["intercept"] + payload["weight"] * normalized


def process_max_memory_mb() -> float:
    """Return the process high-water memory mark in MiB on Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def safe_filename(value: str) -> str:
    """Restrict device identifiers before including them in a directory name."""
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
    local_time = aware_datetime(generated_at).astimezone(config.REPORT_TIMEZONE)
    run_directory = (
        config.REPORT_ROOT
        / "inference_runs"
        / local_time.strftime("%Y-%m-%d")
        / f'{local_time.strftime("%H-%M-%S-%f")}_{safe_filename(inference_motor_id)}'
    )
    csv_directory = run_directory / "csv"
    chart_directory = run_directory / "charts"
    metadata_directory = run_directory / "metadata"
    for directory in (csv_directory, chart_directory, metadata_directory):
        directory.mkdir(parents=True, exist_ok=False)

    training_examples = build_training_examples(training_records)
    evaluation_examples = build_training_examples(inference_records)
    temperature_model = train_temperature_baseline(training_examples, RIDGE_ALPHA)

    five_feature_predictions = [
        predict_with_payload(model_payload, example.features)
        for example in evaluation_examples
    ]
    temperature_predictions = [
        predict_temperature_baseline(temperature_model, float(example.features[0]))
        for example in evaluation_examples
    ]
    persistence_predictions = [
        float(example.features[0]) for example in evaluation_examples
    ]
    actuals = [float(example.target_temperature) for example in evaluation_examples]
    generated_iso = iso_datetime(generated_at)

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
                "來源資料時間": iso_datetime(example.source_time),
                "預測目標時間": iso_datetime(example.target_time),
                "預測溫度_C": round(prediction, 6),
                "30分鐘後實際溫度_C": round(actual, 6),
                "絕對誤差_C": round(abs(error), 6),
                "平方誤差_C2": round(error * error, 6),
                "資料狀態": "歷史回測_已取得真值",
            }
        )

    latest = inference_records[-1]
    latest_target = aware_datetime(latest.recorded_at) + timedelta(
        minutes=FORECAST_HORIZON_MINUTES
    )
    prediction_rows.append(
        {
            "設備編號": inference_motor_id,
            "訓練資料集": training_motor_id,
            "預測產生時間": generated_iso,
            "來源資料時間": iso_datetime(latest.recorded_at),
            "預測目標時間": iso_datetime(latest_target),
            "預測溫度_C": round(latest_prediction, 6),
            "30分鐘後實際溫度_C": "",
            "絕對誤差_C": "",
            "平方誤差_C2": "",
            "資料狀態": "即時推論_等待真值",
        }
    )
    predictions_csv = csv_directory / "predictions.csv"
    write_csv(predictions_csv, tuple(prediction_rows[0]), prediction_rows)

    methods = {
        "Persistence": persistence_predictions,
        "Temperature-only Ridge": temperature_predictions,
        "Five-feature Ridge": five_feature_predictions,
    }
    regression = {
        name: regression_metrics(predictions, actuals)
        for name, predictions in methods.items()
    }
    main_metrics = regression["Five-feature Ridge"]
    metrics_rows = [
        {
            "設備編號": inference_motor_id,
            "訓練資料集": training_motor_id,
            "完成真值樣本數": main_metrics["sample_count"],
            "MAE_C": rounded(main_metrics["mae"]),
            "中位數絕對誤差_C": rounded(main_metrics["median_absolute_error"]),
            "MSE_C2": rounded(main_metrics["mse"]),
            "RMSE_C": rounded(main_metrics["rmse"]),
            "最大絕對誤差_C": rounded(main_metrics["max_error"]),
            "R2": rounded(main_metrics["r2_score"]),
            "MAPE_percent": rounded(main_metrics["mape_percent"]),
            "平均誤差_bias_C": rounded(main_metrics["mean_error"]),
            "異常溫度門檻_C": config.ANOMALY_TEMPERATURE_THRESHOLD,
        }
    ]
    metrics_csv = csv_directory / "metrics_summary.csv"
    write_csv(metrics_csv, tuple(metrics_rows[0]), metrics_rows)

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
                "MAE_C": rounded(metrics["mae"]),
                "中位數絕對誤差_C": rounded(metrics["median_absolute_error"]),
                "MSE_C2": rounded(metrics["mse"]),
                "RMSE_C": rounded(metrics["rmse"]),
                "最大絕對誤差_C": rounded(metrics["max_error"]),
                "R2": rounded(metrics["r2_score"]),
                "MAPE_percent": rounded(metrics["mape_percent"]),
                "相對Persistence改善率_percent": rounded(improvement),
                "備註": "",
            }
        )
    baseline_rows.append(
        {
            "方法": "Static temperature threshold",
            "用途": "傳統異常監控",
            "樣本數": len(actuals),
            "MAE_C": "",
            "中位數絕對誤差_C": "",
            "MSE_C2": "",
            "RMSE_C": "",
            "最大絕對誤差_C": "",
            "R2": "",
            "MAPE_percent": "",
            "相對Persistence改善率_percent": "",
            "備註": "門檻法為分類器，不以回歸誤差評估",
        }
    )
    baseline_csv = csv_directory / "baseline_comparison.csv"
    write_csv(baseline_csv, tuple(baseline_rows[0]), baseline_rows)

    actual_anomalies = [
        actual >= config.ANOMALY_TEMPERATURE_THRESHOLD for actual in actuals
    ]
    anomaly_score_methods = {
        **methods,
        "Static temperature threshold": [
            float(example.features[0]) for example in evaluation_examples
        ],
    }
    anomaly_rows = []
    for name, scores in anomaly_score_methods.items():
        predicted_anomalies = [
            value >= config.ANOMALY_TEMPERATURE_THRESHOLD for value in scores
        ]
        metrics = classification_metrics(
            predicted_anomalies,
            actual_anomalies,
            FORECAST_HORIZON_MINUTES,
        )
        score_metrics = binary_score_metrics(scores, actual_anomalies)
        anomaly_rows.append(
            {
                "方法": name,
                "異常定義": (
                    "30分鐘後溫度 >= "
                    f"{config.ANOMALY_TEMPERATURE_THRESHOLD} C"
                ),
                "真值來源": "目標時間實際溫度；不使用人工輸入的status欄位",
                **metrics,
                **score_metrics,
            }
        )
    anomaly_csv = csv_directory / "anomaly_detection.csv"
    write_csv(anomaly_csv, tuple(anomaly_rows[0]), anomaly_rows)

    source_labels = [
        aware_datetime(example.source_time)
        .astimezone(config.REPORT_TIMEZONE)
        .strftime("%H:%M")
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
    write_line_chart(
        actual_chart,
        "Actual vs Predicted Temperature",
        (
            ("Actual", actuals, "#0f766e"),
            ("Predicted", five_feature_predictions, "#2563eb"),
        ),
        source_labels,
        "Temperature (C)",
    )
    write_line_chart(
        error_chart,
        "Prediction Error by Observation",
        (("Prediction - Actual", signed_errors, "#dc2626"),),
        source_labels,
        "Error (C)",
    )
    write_histogram(
        distribution_chart,
        "Prediction Error Distribution",
        signed_errors,
    )
    regression_chart_rows = [
        (name, metrics["mae"])
        for name, metrics in regression.items()
        if metrics["mae"] is not None
    ]
    write_bar_chart(
        baseline_chart,
        "Forecast Baseline MAE Comparison",
        [name for name, _ in regression_chart_rows],
        [value for _, value in regression_chart_rows],
        "MAE (C)",
    )
    write_bar_chart(
        anomaly_chart,
        "Anomaly Detection F1 Comparison",
        [row["方法"] for row in anomaly_rows],
        [row["f1_score"] for row in anomaly_rows],
        "F1 score",
    )
    write_bar_chart(
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
        not is_finite(getattr(record, field, None))
        for record in inference_records
        for field in ("temperature", "humidity", "accel_x", "accel_y", "accel_z")
    )
    ordered_times = sorted(
        aware_datetime(record.recorded_at)
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
    system_rows = [
        {
            "模型訓練時間_ms": rounded(model_payload.get("training_duration_ms")),
            "單次模型推論時間_ms": round(model_inference_duration_ms, 6),
            "Agent工具呼叫正確率": "尚未建立人工標註集",
            "Agent回答與資料一致率": "尚未建立人工標註集",
            "程序CPU時間_ms": round(process_cpu_time_ms, 6),
            "程序最大RAM_MB": round(process_max_memory_mb(), 3),
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
                time.monotonic() - config.PROCESS_STARTED_AT,
                3,
            ),
        }
    ]
    system_csv = csv_directory / "system_performance.csv"
    write_csv(system_csv, tuple(system_rows[0]), system_rows)
    write_bar_chart(
        system_chart,
        "System Performance Duration",
        ("Training", "Inference"),
        (
            float(model_payload.get("training_duration_ms") or 0),
            model_inference_duration_ms,
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
        "regression_baselines": regression,
        "anomaly_classification": {
            row["方法"]: {
                key: value
                for key, value in row.items()
                if key not in {"方法", "異常定義", "真值來源"}
            }
            for row in anomaly_rows
        },
        "anomaly_temperature_threshold_c": (
            config.ANOMALY_TEMPERATURE_THRESHOLD
        ),
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
    (config.REPORT_ROOT / "latest_run.txt").write_text(
        str(run_directory),
        encoding="utf-8",
    )
    return summary
