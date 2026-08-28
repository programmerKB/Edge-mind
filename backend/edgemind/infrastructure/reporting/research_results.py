"""Durable JSON/CSV artifacts for reproducible forecasting experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


EXPERIMENT_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
BULKY_LEDGER_FIELDS = frozenset({"prediction_records", "split_manifest_records"})


def _json_ready(value: Any) -> Any:
    """Reject non-finite numbers and normalize tuples before durable output."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _without_bulk_ledgers(value: Any) -> Any:
    """Remove per-sample rows after their dedicated CSV artifacts are written."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_bulk_ledgers(item)
            for key, item in value.items()
            if str(key) not in BULKY_LEDGER_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_without_bulk_ledgers(item) for item in value]
    return value


def _flatten_scalars(
    value: Any,
    *,
    prefix: str = "",
    excluded: tuple[str, ...] = (
        "predictions",
        "by_horizon",
        "folds",
        "latest_forecast",
        "trajectory",
    ),
) -> dict[str, Any]:
    """Flatten scalar metric leaves while excluding bulky prediction arrays."""
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in excluded:
                continue
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            flattened.update(
                _flatten_scalars(child, prefix=child_prefix, excluded=excluded)
            )
    elif isinstance(value, (str, int, float, bool)) or value is None:
        flattened[prefix or "value"] = value
    return flattened


def _named_items(value: Any, name_field: str) -> list[tuple[str, Mapping]]:
    if isinstance(value, Mapping):
        return [
            (str(name), payload if isinstance(payload, Mapping) else {"value": payload})
            for name, payload in value.items()
        ]
    if isinstance(value, list):
        return [
            (
                str(item.get(name_field) or item.get("name") or index),
                item,
            )
            for index, item in enumerate(value)
            if isinstance(item, Mapping)
        ]
    return []


def _write_table(path: Path, rows: Iterable[dict]) -> bool:
    rows = list(rows)
    if not rows:
        return False
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return True


def _model_rows(result: Mapping) -> list[dict]:
    return [
        {"model": name, **_flatten_scalars(payload)}
        for name, payload in _named_items(result.get("models"), "model")
    ]


def _horizon_rows(result: Mapping) -> list[dict]:
    rows: list[dict] = []
    for model_name, model in _named_items(result.get("models"), "model"):
        for split_name in ("validation", "test", "cross_device"):
            evaluation = model.get(split_name)
            if not isinstance(evaluation, Mapping):
                continue
            horizons = evaluation.get("by_horizon") or evaluation.get("horizons")
            for horizon, metrics in _named_items(horizons, "horizon_minutes"):
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "horizon_minutes": horizon,
                        **_flatten_scalars(metrics),
                    }
                )
    return rows


def _ablation_rows(result: Mapping) -> list[dict]:
    ablations = result.get("ablations")
    if isinstance(ablations, Mapping) and isinstance(ablations.get("results"), (list, Mapping)):
        ablations = ablations["results"]
    return [
        {"feature_set": name, **_flatten_scalars(payload)}
        for name, payload in _named_items(ablations, "feature_set")
    ]


def _prediction_rows(result: Mapping) -> list[dict]:
    rows: list[dict] = []
    for model_name, model in _named_items(result.get("models"), "model"):
        for split_name in ("validation", "test", "cross_device"):
            evaluation = model.get(split_name)
            if not isinstance(evaluation, Mapping):
                continue
            records = evaluation.get("prediction_records")
            if not isinstance(records, list):
                continue
            rows.extend(
                {
                    "model": model_name,
                    "split": split_name,
                    **record,
                }
                for record in records
                if isinstance(record, Mapping)
            )
    return rows


def _split_manifest_rows(result: Mapping) -> list[dict]:
    dataset = result.get("dataset")
    if not isinstance(dataset, Mapping):
        return []
    records = dataset.get("split_manifest_records")
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _risk_event_rows(result: Mapping) -> list[dict]:
    def event_record(record: Mapping) -> dict:
        flattened = _flatten_scalars(record)
        for key, value in record.items():
            if isinstance(value, (list, tuple)):
                flattened[str(key)] = json.dumps(value, ensure_ascii=False)
        return flattened

    rows: list[dict] = []
    for model_name, model in _named_items(result.get("models"), "model"):
        for split_name in ("validation", "test", "cross_device"):
            evaluation = model.get(split_name)
            if not isinstance(evaluation, Mapping):
                continue
            event_evaluation = evaluation.get("risk", {}).get("event_evaluation")
            if not isinstance(event_evaluation, Mapping):
                continue
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "record_type": "summary",
                    **_flatten_scalars(event_evaluation),
                }
            )
            for collection, record_type in (
                (event_evaluation.get("events", {}).get("records"), "event"),
                (event_evaluation.get("warnings", {}).get("records"), "warning"),
                (event_evaluation.get("matches"), "match"),
            ):
                if not isinstance(collection, list):
                    continue
                rows.extend(
                    {
                        "model": model_name,
                        "split": split_name,
                        "record_type": record_type,
                        **event_record(record),
                    }
                    for record in collection
                    if isinstance(record, Mapping)
                )
    return rows


def _statistical_rows(result: Mapping) -> list[dict]:
    comparisons = result.get("statistical_comparisons")
    if not isinstance(comparisons, list):
        return []
    return [
        _flatten_scalars(item)
        for item in comparisons
        if isinstance(item, Mapping)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_research_experiment(result: dict, report_root: Path) -> dict:
    """Atomically persist a completed result plus analysis-friendly CSV tables."""
    experiment_id = str(result.get("experiment_id", ""))
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("研究實驗 ID 格式不合法")
    directory = report_root / "research_experiments" / experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    files = {"result_json": str(directory / "result.json")}
    tables = (
        ("model_comparison_csv", directory / "model_comparison.csv", _model_rows(result)),
        ("horizon_metrics_csv", directory / "horizon_metrics.csv", _horizon_rows(result)),
        ("feature_ablations_csv", directory / "feature_ablations.csv", _ablation_rows(result)),
        ("predictions_csv", directory / "predictions.csv", _prediction_rows(result)),
        (
            "split_manifest_csv",
            directory / "split_manifest.csv",
            _split_manifest_rows(result),
        ),
        ("risk_events_csv", directory / "risk_events.csv", _risk_event_rows(result)),
        (
            "statistical_comparisons_csv",
            directory / "statistical_comparisons.csv",
            _statistical_rows(result),
        ),
    )
    checksums: dict[str, str] = {}
    for key, path, rows in tables:
        if _write_table(path, rows):
            files[key] = str(path)
            checksums[path.name] = _sha256(path)

    compact_result = _without_bulk_ledgers(result)
    enriched = _json_ready(
        {
            **compact_result,
            "artifacts": {
                **(compact_result.get("artifacts") or {}),
                "run_directory": str(directory),
                **files,
                "sha256": checksums,
            },
        }
    )
    result_path = directory / "result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(result_path)
    return enriched


def get_research_experiment(experiment_id: str, report_root: Path) -> dict | None:
    """Load one allow-listed result path without permitting traversal."""
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        return None
    path = report_root / "research_experiments" / experiment_id / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_research_forecast(result: dict, report_root: Path) -> dict:
    """Persist an immutable live forecast separately from offline experiments."""
    forecast_id = str(result.get("forecast_id", ""))
    if not EXPERIMENT_ID_PATTERN.fullmatch(forecast_id):
        raise ValueError("研究預測 ID 格式不合法")
    directory = report_root / "research_forecasts" / forecast_id
    directory.mkdir(parents=True, exist_ok=False)
    result_path = directory / "result.json"
    enriched = _json_ready(
        {
            **result,
            "artifacts": {
                **(result.get("artifacts") or {}),
                "run_directory": str(directory),
                "result_json": str(result_path),
            },
        }
    )
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(result_path)
    return enriched


def get_research_forecast(forecast_id: str, report_root: Path) -> dict | None:
    """Load one traversal-safe, immutable live forecast result."""
    if not EXPERIMENT_ID_PATTERN.fullmatch(forecast_id):
        return None
    path = report_root / "research_forecasts" / forecast_id / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
