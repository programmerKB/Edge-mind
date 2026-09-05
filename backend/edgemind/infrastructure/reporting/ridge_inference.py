"""Evaluate live Ridge variants using their exact history windows and weights."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from edgemind.domain.entities import SensorReading
from edgemind.domain.forecasting import TrainingExample
from edgemind.domain.ridge_experiments import (
    DIRECT_MODEL,
    RidgeParameters,
    build_experiment_examples,
    predict_ridge,
)
from edgemind.infrastructure.reporting.context import ReportContext
from edgemind.infrastructure.reporting.inference import (
    predict_temperature_baseline,
    train_temperature_baseline,
    write_inference_report,
)


def create_ridge_inference_report(
    *,
    context: ReportContext,
    training_records: Sequence[SensorReading],
    inference_records: Sequence[SensorReading],
    parameters: RidgeParameters,
    forecast: dict,
    training_duration_ms: float,
    model_inference_duration_ms: float,
    process_cpu_time_ms: float,
) -> dict:
    """Use the selected live model for backtests, separately from held-out metrics."""
    training = build_experiment_examples(training_records)
    evaluation = build_experiment_examples(inference_records)

    def report_examples(examples):
        return [
            TrainingExample(
                source_time=example.origin_time,
                target_time=example.target_time,
                features=example.direct_features,
                target_temperature=example.target_temperature,
            )
            for example in examples
        ]

    temperature_model = train_temperature_baseline(
        report_examples(training), forecast["selected_alpha"]
    )
    methods = {
        "Persistence": [example.current_temperature for example in evaluation],
        "Temperature-only Ridge": [
            predict_temperature_baseline(temperature_model, example.current_temperature)
            for example in evaluation
        ],
        forecast["model_label"]: [
            example.current_temperature + predict_ridge(
                parameters,
                example.direct_features
                if forecast["model_name"] == DIRECT_MODEL
                else example.history_features,
            )
            for example in evaluation
        ],
    }
    return write_inference_report(
        context=context,
        inference_records=inference_records,
        evaluation_examples=report_examples(evaluation),
        methods=methods,
        main_method=forecast["model_label"],
        model_payload={
            key: forecast[key]
            for key in (
                "model_name", "model_label", "selected_alpha", "feature_count",
                "history_minutes", "training_sample_count", "validation_metrics",
                "test_metrics",
            )
        } | {"training_duration_ms": training_duration_ms},
        inference_motor_id=forecast["motor_id"],
        training_motor_id=forecast["training_motor_id"],
        latest_prediction=forecast["predicted_temperature"],
        latest_source_time=datetime.fromisoformat(forecast["source_recorded_at"]),
        generated_at=datetime.fromisoformat(forecast["generated_at"]),
        model_inference_duration_ms=model_inference_duration_ms,
        process_cpu_time_ms=process_cpu_time_ms,
    )
