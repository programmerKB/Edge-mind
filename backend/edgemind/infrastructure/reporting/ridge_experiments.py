"""Human-readable filesystem output for dual-Ridge experiments."""

from __future__ import annotations

import json
from datetime import datetime
import re

from edgemind.infrastructure.reporting.context import ReportContext
from edgemind.infrastructure.reporting.io import write_csv


EXPERIMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _metric_row(result: dict, model: dict) -> dict:
    validation = model["validation_metrics"]
    testing = model["test_metrics"]
    external = model.get("external_metrics", {})
    return {
        "experiment_id": result["experiment_id"],
        "training_motor_id": result["training_motor_id"],
        "evaluation_motor_id": result.get("evaluation_motor_id") or "",
        "model_name": model["model_name"],
        "model_label": model["label"],
        "feature_count": model["feature_count"],
        "selected_alpha": model["selected_alpha"],
        "validation_mae": validation["mae"],
        "validation_rmse": validation["rmse"],
        "test_sample_count": testing["sample_count"],
        "test_mae": testing["mae"],
        "test_rmse": testing["rmse"],
        "test_r2": testing["r2_score"],
        "external_sample_count": external.get("sample_count", ""),
        "external_mae": external.get("mae", ""),
        "external_rmse": external.get("rmse", ""),
        "external_r2": external.get("r2_score", ""),
    }


def save_ridge_experiment(context: ReportContext, result: dict) -> dict:
    """Save JSON plus comparison and prediction CSVs in a dated directory."""
    created_at = result["created_at"]
    local_time = datetime.fromisoformat(created_at).astimezone(context.timezone)
    run_name = f"{local_time:%H-%M-%S-%f}_{result['experiment_id'][:8]}"
    run_directory = (
        context.root / "ridge_experiments" / f"{local_time:%Y-%m-%d}" / run_name
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    predictions = result.get("predictions", [])
    public_result = {
        key: value for key, value in result.items() if key != "predictions"
    }
    relative_directory = run_directory.relative_to(context.root).as_posix()
    public_result["artifacts"] = {
        "run_directory": relative_directory,
        "result_json": f"{relative_directory}/result.json",
        "model_comparison_csv": f"{relative_directory}/model_comparison.csv",
        "test_predictions_csv": f"{relative_directory}/test_predictions.csv",
    }
    (run_directory / "result.json").write_text(
        json.dumps(public_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        run_directory / "model_comparison.csv",
        (
            "experiment_id",
            "training_motor_id",
            "evaluation_motor_id",
            "model_name",
            "model_label",
            "feature_count",
            "selected_alpha",
            "validation_mae",
            "validation_rmse",
            "test_sample_count",
            "test_mae",
            "test_rmse",
            "test_r2",
            "external_sample_count",
            "external_mae",
            "external_rmse",
            "external_r2",
        ),
        [_metric_row(result, model) for model in result["models"]],
    )
    write_csv(
        run_directory / "test_predictions.csv",
        (
            "model_name",
            "origin_time",
            "target_time",
            "actual_temperature",
            "predicted_temperature",
            "error",
        ),
        predictions,
    )
    return public_result


def get_ridge_experiment(
    context: ReportContext,
    experiment_id: str,
) -> dict | None:
    """Find one validated experiment identifier below the report root."""
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        return None
    base = context.root / "ridge_experiments"
    if not base.is_dir():
        return None
    matches = list(base.glob(f"*/*_{experiment_id[:8]}/result.json"))
    for path in matches:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment_id") == experiment_id:
            return payload
    return None
