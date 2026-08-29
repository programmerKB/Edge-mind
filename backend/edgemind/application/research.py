"""Use cases for reproducible multi-horizon forecasting experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Sequence
from uuid import uuid4

from edgemind.application.ports import ResearchResultGateway, UnitOfWork
from edgemind.domain.entities import SensorReading
from edgemind.domain.research import (
    BASE_FEATURE_NAMES,
    DEFAULT_HORIZONS_MINUTES,
    DEFAULT_MODEL_NAMES,
    ModelRegistry,
    ResearchConfig,
    ResearchError,
    build_inference_sequence,
    build_sequence_dataset,
    chronological_split,
    derive_risk,
    evaluate_model,
    run_research_experiment,
)


ALL_RESEARCH_MODELS = DEFAULT_MODEL_NAMES
MIN_THRESHOLD_C = 20.0
MAX_THRESHOLD_C = 120.0
HISTORICAL_CHART_POINT_LIMIT = 60


def _validated_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ResearchError("threshold_c must be finite")
    if not MIN_THRESHOLD_C <= threshold <= MAX_THRESHOLD_C:
        raise ResearchError(
            f"threshold_c must be between {MIN_THRESHOLD_C:g} and "
            f"{MAX_THRESHOLD_C:g} °C for this application contract"
        )
    return threshold


def _reading_hash(records: Sequence[SensorReading]) -> str:
    """Fingerprint exactly the values and ordering supplied to an experiment."""
    digest = hashlib.sha256()
    for record in records:
        row = [
            record.motor_id,
            record.recorded_at.isoformat() if record.recorded_at else None,
            record.temperature,
            record.humidity,
            record.accel_x,
            record.accel_y,
            record.accel_z,
            record.status,
        ]
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _synthetic_evidence(records: Sequence[SensorReading]) -> bool:
    """Recognize only generator/demo labels; unknown data is never relabelled."""
    labels = [str(record.status or "").lower() for record in records]
    return any(
        label.startswith(("demo", "synthetic")) for label in labels
    )


def _fit_with_validation(model, training, validation) -> None:
    """Select supported training controls without exposing locked-test labels."""
    method = getattr(model, "fit_with_validation", None)
    if callable(method):
        method(training, validation)
    else:
        model.fit(training)


def _refit(model, examples) -> None:
    """Refit with frozen training controls on a larger known-history release."""
    method = getattr(model, "refit_on_development", None)
    if callable(method):
        method(examples)
    else:
        model.fit(examples)


def _compact_historical_evaluation(evaluation: dict) -> dict:
    """Keep auditable metrics and a bounded time series without bulky ledgers."""
    prediction_records = [
        record
        for record in evaluation.get("prediction_records", [])
        if isinstance(record, dict)
    ]
    chart_horizon = max(
        (
            int(record["horizon_minutes"])
            for record in prediction_records
            if isinstance(record.get("horizon_minutes"), (int, float))
        ),
        default=None,
    )
    chart_records = (
        [
            record
            for record in prediction_records
            if int(record.get("horizon_minutes", -1)) == chart_horizon
        ][-HISTORICAL_CHART_POINT_LIMIT:]
        if chart_horizon is not None
        else []
    )
    return {
        "truth_status": "observed",
        "overall": evaluation["overall"],
        "target_distribution": evaluation["target_distribution"],
        "by_horizon": evaluation["by_horizon"],
        "latest_forecast": evaluation["latest_forecast"],
        "chart_series": {
            "horizon_minutes": chart_horizon,
            "point_count": len(chart_records),
            "points": [
                {
                    "origin_time": record.get("origin_time"),
                    "target_time": record.get("target_time"),
                    "actual_temperature_c": record.get("actual"),
                    "predicted_temperature_c": record.get("predicted"),
                }
                for record in chart_records
            ],
        },
        "inference_latency_ms": evaluation["inference_latency_ms"],
    }


class ResearchService:
    """Coordinate sensor repositories, pure study rules, and result storage."""

    def __init__(self, reports: ResearchResultGateway, registry: ModelRegistry):
        self._reports = reports
        self._registry = registry

    def configuration(self, uow: UnitOfWork) -> dict:
        """Expose one canonical UI/API contract and current dataset inventory."""
        summaries = uow.sensors.list_device_summaries()
        for summary in summaries:
            count = int(summary.get("complete_reading_count", 0))
            summary["estimated_complete_sequences"] = max(0, count - 12 - 6 + 1)
            summary["pipeline_eligible"] = count >= 36
            summary["pipeline_eligibility_scope"] = (
                "default_60min_history_30min_horizon_count_precheck"
            )
            summary["runtime_quality_audit_required"] = True
            summary["study_eligible"] = count >= 2_016
            summary["study_eligibility_reason"] = (
                "至少七天的五分鐘資料，可開始探索性研究"
                if summary["study_eligible"]
                else "資料少於七天（2,016 筆），僅建議流程驗證"
            )

        identifiers = [summary["motor_id"] for summary in summaries]
        training_default = (
            "RESEARCH-A"
            if "RESEARCH-A" in identifiers
            else "DEMO-1" if "DEMO-1" in identifiers else (identifiers[0] if identifiers else "")
        )
        evaluation_default = (
            "RESEARCH-B"
            if "RESEARCH-B" in identifiers
            else "DEMO-2" if "DEMO-2" in identifiers else (
                next((item for item in identifiers if item != training_default), training_default)
            )
        )
        defaults = ResearchConfig().to_dict()
        defaults.update(
            {
                "training_motor_id": training_default,
                "evaluation_motor_id": evaluation_default,
                "model_names": list(ALL_RESEARCH_MODELS),
                "include_ablations": True,
            }
        )
        return {
            "schema_version": 1,
            "study_title": "多感測器設備溫度軌跡與過熱風險預測",
            "defaults": defaults,
            "motors": summaries,
            "models": self._registry.describe(),
            "constraints": {
                "exact_sampling_required": True,
                "allowed_horizons_minutes": list(range(5, 61, 5)),
                "minimum_gap_steps": 6,
                "minimum_pipeline_readings": 36,
                "minimum_pipeline_readings_formula": (
                    "smallest N satisfying sequence_count=N-history_steps-"
                    "max_horizon_steps+1, two horizon gaps, chronological "
                    "60/20/20, and development-only walk-forward folds"
                ),
                "minimum_exploratory_readings": 2_016,
                "recommended_formal_readings": 8_640,
                "recommended_formal_duration_days": 30,
                "synthetic_v2_devices": 6,
                "synthetic_v2_duration_days": 90,
                "synthetic_v2_total_readings": 155_520,
                "synthetic_data_research_claims_allowed": False,
            },
            "warnings": [
                "DEMO 與 RESEARCH 合成資料只驗證流程，不代表真實設備效能。",
                "正式結論必須使用預先登記、具設備/工況來源的長期真實資料。",
                "所有模型使用完全相同的時間分割、gap、horizon 與特徵消融規則。",
                "模型只可用 development validation 選超參數；locked test 不參與調參。",
                "低變異 holdout 的 R² 可能非常負，須與 MAE 及 Direct Ridge skill 一起解讀。",
            ],
        }

    def run(
        self,
        uow: UnitOfWork,
        *,
        training_motor_id: str,
        evaluation_motor_id: str | None,
        history_minutes: int = 60,
        horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
        threshold_c: float = 35.0,
        model_names: Sequence[str] = ALL_RESEARCH_MODELS,
        include_ablations: bool = True,
        sampling_minutes: int = 5,
        train_fraction: float = 0.60,
        validation_fraction: float = 0.20,
        gap_steps: int | None = None,
        walk_forward_folds: int = 3,
        random_seed: int = 42,
    ) -> dict:
        """Run synchronously and persist only genuine computed measurements."""
        training_motor_id = training_motor_id.strip()
        evaluation_motor_id = (
            evaluation_motor_id.strip() if evaluation_motor_id else None
        )
        if not training_motor_id:
            raise ResearchError("training_motor_id cannot be empty")
        if sampling_minutes <= 0:
            raise ResearchError("sampling_minutes must be positive")
        if history_minutes % sampling_minutes:
            raise ResearchError("history_minutes must be divisible by sampling_minutes")
        if history_minutes < sampling_minutes * 2:
            raise ResearchError("history_minutes must include at least two readings")
        threshold_c = _validated_threshold(threshold_c)
        config = ResearchConfig(
            sampling_minutes=sampling_minutes,
            history_steps=history_minutes // sampling_minutes,
            horizons_minutes=tuple(int(value) for value in horizons_minutes),
            threshold_c=threshold_c,
            train_fraction=float(train_fraction),
            validation_fraction=float(validation_fraction),
            gap_steps=gap_steps,
            walk_forward_folds=walk_forward_folds,
            random_seed=random_seed,
        )
        training_records = uow.sensors.list_readings(training_motor_id)
        if not training_records:
            raise ResearchError(f"找不到訓練設備 {training_motor_id} 的感測資料")
        evaluation_records = None
        if evaluation_motor_id and evaluation_motor_id != training_motor_id:
            evaluation_records = uow.sensors.list_readings(evaluation_motor_id)
            if not evaluation_records:
                raise ResearchError(f"找不到評估設備 {evaluation_motor_id} 的感測資料")

        normalized_model_names = tuple(
            str(model_name).strip().lower() for model_name in model_names
        )
        unsupported_models = sorted(
            set(normalized_model_names) - set(ALL_RESEARCH_MODELS)
        )
        if unsupported_models:
            raise ResearchError(
                "unsupported model_names: "
                f"{', '.join(unsupported_models)}; allowed models: "
                f"{', '.join(ALL_RESEARCH_MODELS)}"
            )

        result = run_research_experiment(
            training_records,
            evaluation_records,
            config=config,
            model_names=normalized_model_names,
            registry=self._registry,
            include_ablations=include_ablations,
        )
        source_is_synthetic = _synthetic_evidence(training_records) or (
            evaluation_records is not None and _synthetic_evidence(evaluation_records)
        )
        provenance_status = (
            "synthetic_or_demo" if source_is_synthetic else "unverified"
        )
        completed = {
            "experiment_id": uuid4().hex,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_provenance": {
                "training_motor_id": training_motor_id,
                "evaluation_motor_id": evaluation_motor_id,
                "training_reading_count": len(training_records),
                "evaluation_reading_count": (
                    len(evaluation_records) if evaluation_records is not None else 0
                ),
                "training_sha256": _reading_hash(training_records),
                "evaluation_sha256": (
                    _reading_hash(evaluation_records)
                    if evaluation_records is not None
                    else None
                ),
                "synthetic_or_demo_detected": source_is_synthetic,
                "provenance_status": provenance_status,
                # The legacy sensor table has no signed/frozen dataset
                # manifest. Absence of a synthetic label is not proof of an
                # audited real source, so the workbench must fail closed.
                "research_claims_allowed": False,
                "provenance_audit_required": True,
                "evidence_note": (
                    "合成/Demo 資料僅可驗證系統流程，不可作為真實效能結論。"
                    if source_is_synthetic
                    else "資料未連結經驗證的不可變來源 manifest；只能作探索與流程驗證。"
                ),
            },
            **result,
        }
        return self._reports.save_research_experiment(completed)

    def get(self, experiment_id: str) -> dict | None:
        """Return one durable completed result."""
        return self._reports.get_research_experiment(experiment_id)

    def forecast_trajectory(
        self,
        uow: UnitOfWork,
        *,
        motor_id: str,
        training_motor_id: str | None = None,
        model_name: str = "ridge_history_trend",
        history_minutes: int = 60,
        horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
        threshold_c: float = 35.0,
        sampling_minutes: int = 5,
        random_seed: int = 42,
    ) -> dict:
        """Fit on known historical labels and forecast the newest pending future."""
        motor_id = motor_id.strip()
        training_motor_id = (training_motor_id or motor_id).strip()
        model_name = model_name.strip().lower()
        if not motor_id or not training_motor_id:
            raise ResearchError("motor_id and training_motor_id cannot be empty")
        if model_name not in ALL_RESEARCH_MODELS:
            raise ResearchError(
                f"unsupported model_name: {model_name}; allowed models: "
                f"{', '.join(ALL_RESEARCH_MODELS)}"
            )
        threshold_c = _validated_threshold(threshold_c)
        if sampling_minutes <= 0:
            raise ResearchError("sampling_minutes must be positive")
        if history_minutes % sampling_minutes:
            raise ResearchError("history_minutes must be divisible by sampling_minutes")
        config = ResearchConfig(
            sampling_minutes=sampling_minutes,
            history_steps=history_minutes // sampling_minutes,
            horizons_minutes=tuple(int(value) for value in horizons_minutes),
            threshold_c=threshold_c,
            random_seed=random_seed,
        )
        availability = self._registry.availability(model_name)
        registration = self._registry.get(model_name)
        if availability["status"] != "available" or registration is None:
            reason = availability.get("reason") or "model is unavailable"
            raise ResearchError(f"{model_name}: {reason}")
        if registration.factory is None:
            raise ResearchError(f"{model_name}: model factory is unavailable")

        training_records = uow.sensors.list_readings(training_motor_id)
        inference_records = (
            training_records
            if motor_id == training_motor_id
            else uow.sensors.list_readings(motor_id)
        )
        if not training_records:
            raise ResearchError(f"找不到訓練設備 {training_motor_id} 的感測資料")
        if not inference_records:
            raise ResearchError(f"找不到推論設備 {motor_id} 的感測資料")
        training_dataset = build_sequence_dataset(training_records, config)
        if not training_dataset.examples:
            raise ResearchError(
                f"設備 {training_motor_id} 沒有完整的 12×5 history 與多 horizon 標籤"
            )
        inference = build_inference_sequence(
            inference_records,
            config,
            device_id=motor_id,
        )
        latest_training_target = max(
            timestamp
            for example in training_dataset.examples
            for timestamp in example.target_times
        )
        if latest_training_target > inference.anchor_time:
            raise ResearchError(
                "training labels extend beyond the forecast origin; "
                "use a time-valid training release"
            )

        split = chronological_split(training_dataset, config)
        development_examples = tuple(
            example
            for example in training_dataset.examples
            if example.anchor_time <= split.validation[-1].anchor_time
        )
        if any(
            target_time >= split.test[0].anchor_time
            for example in development_examples
            for target_time in example.target_times
        ):
            raise ResearchError(
                "development target crosses the locked-test boundary"
            )

        model = registration.factory(config, BASE_FEATURE_NAMES)
        evaluation_training_started = time.perf_counter_ns()
        _fit_with_validation(model, split.train, split.validation)
        validation_evaluation = evaluate_model(model, split.validation, config)
        _refit(model, development_examples)
        locked_test_evaluation = evaluate_model(model, split.test, config)
        evaluation_training_time_ms = (
            time.perf_counter_ns() - evaluation_training_started
        ) / 1_000_000

        final_refit_started = time.perf_counter_ns()
        _refit(model, training_dataset.examples)
        final_refit_time_ms = (
            time.perf_counter_ns() - final_refit_started
        ) / 1_000_000
        training_time_ms = evaluation_training_time_ms + final_refit_time_ms
        inference_started = time.perf_counter_ns()
        predictions = tuple(float(value) for value in model.predict(inference))
        inference_time_ms = (time.perf_counter_ns() - inference_started) / 1_000_000
        if len(predictions) != len(config.horizons_minutes):
            raise ResearchError("model output width does not match horizons")
        if not all(math.isfinite(value) for value in predictions):
            raise ResearchError("model returned non-finite trajectory values")
        risk = derive_risk(
            predictions,
            config.horizons_minutes,
            threshold_c=config.threshold_c,
            current_temperature=inference.current_temperature,
            medium_margin_c=config.medium_risk_margin_c,
            rapid_heating_c_per_minute=config.rapid_heating_c_per_minute,
        )
        state = model.state_dict()
        model_size_bytes = len(
            json.dumps(
                state,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        parameter_method = getattr(model, "parameter_count", None)
        parameter_count = int(parameter_method()) if callable(parameter_method) else None
        source_is_synthetic = _synthetic_evidence(training_records) or _synthetic_evidence(
            inference_records
        )
        provenance_status = (
            "synthetic_or_demo" if source_is_synthetic else "unverified"
        )
        result = {
            "schema_version": 1,
            "forecast_id": uuid4().hex,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "truth_status": "pending",
            "configuration": config.to_dict(),
            "motor_id": motor_id,
            "training_motor_id": training_motor_id,
            "model": {
                **availability,
                "representation": getattr(model, "representation", "custom_adapter"),
                "feature_names": list(getattr(model, "feature_names", ())),
                "training_sequence_count": len(training_dataset.examples),
                "training_time_ms": training_time_ms,
                "inference_time_ms": inference_time_ms,
                "model_size_bytes": model_size_bytes,
                "parameter_count": parameter_count,
            },
            "source": inference.to_dict(),
            "trajectory": [
                {
                    "horizon_minutes": horizon,
                    "target_time": inference.target_times[index].isoformat(),
                    "predicted_temperature_c": predictions[index],
                    "truth_status": "pending",
                }
                for index, horizon in enumerate(config.horizons_minutes)
            ],
            "risk": risk.to_dict(),
            "historical_evaluation": {
                "status": "completed",
                "protocol": "chronological_train_validation_locked_test",
                "description": (
                    "模型誤差來自已有真值的歷史鎖定測試集；即時未來預測"
                    "則等待目標時間到達後另行驗證。"
                ),
                "split": split.to_dict(config.sampling_minutes),
                "validation": _compact_historical_evaluation(
                    validation_evaluation
                ),
                "locked_test": _compact_historical_evaluation(
                    locked_test_evaluation
                ),
                "locked_test_used_for_selection": False,
                "final_operational_refit_sample_count": len(
                    training_dataset.examples
                ),
                "evaluation_training_time_ms": evaluation_training_time_ms,
                "final_refit_time_ms": final_refit_time_ms,
            },
            "training_dataset": training_dataset.to_dict(),
            "leakage_audit": {
                "latest_training_target_time": latest_training_target.isoformat(),
                "forecast_origin_time": inference.anchor_time.isoformat(),
                "training_labels_do_not_follow_forecast_origin": True,
                "future_truth_used_for_prediction": False,
            },
            "dataset_provenance": {
                "training_sha256": _reading_hash(training_records),
                "inference_sha256": _reading_hash(inference_records),
                "synthetic_or_demo_detected": source_is_synthetic,
                "provenance_status": provenance_status,
                "research_claims_allowed": False,
                "provenance_audit_required": True,
            },
        }
        return self._reports.save_research_forecast(result)

    def get_forecast(self, forecast_id: str) -> dict | None:
        """Return one immutable pending-truth trajectory forecast."""
        return self._reports.get_research_forecast(forecast_id)


__all__ = ["ALL_RESEARCH_MODELS", "ResearchService"]
