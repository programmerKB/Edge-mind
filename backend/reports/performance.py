"""Persistent aggregation of system performance across inference runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from threading import Lock
from typing import Any

from evaluation import descriptive_statistics


_HISTORY_LOCK = Lock()
HISTORY_FIELDS = (
    "run_id",
    "generated_at",
    "inference_motor_id",
    "training_motor_id",
    "model_training_duration_ms",
    "model_inference_duration_ms",
    "process_cpu_time_ms",
    "process_max_ram_mb",
    "feature_missing_rate_percent",
    "reading_missing_rate_percent",
    "uptime_seconds",
)
NUMERIC_FIELDS = HISTORY_FIELDS[4:]
SYSTEM_FIELD_MAP = {
    "model_training_duration_ms": "模型訓練時間_ms",
    "model_inference_duration_ms": "單次模型推論時間_ms",
    "process_cpu_time_ms": "程序CPU時間_ms",
    "process_max_ram_mb": "程序最大RAM_MB",
    "feature_missing_rate_percent": "感測特徵遺失率_percent",
    "reading_missing_rate_percent": "感測資料筆數遺失率_percent",
    "uptime_seconds": "系統連續運作時間_seconds",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV or return an empty history shape when it does not exist."""
    if not path.is_file():
        return list(HISTORY_FIELDS), []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or HISTORY_FIELDS), list(reader)


def _write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict]) -> None:
    """Atomically replace a performance CSV under the caller-held lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _number(value: Any) -> float | None:
    """Convert populated CSV cells to floats while preserving blanks."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def record_performance_run(system_csv_path: str) -> dict:
    """Upsert one completed run and rebuild aggregate CSV/JSON summaries."""
    system_path = Path(system_csv_path)
    run_directory = system_path.parent.parent
    report_root = run_directory.parents[2]
    metadata_path = run_directory / "metadata" / "run_summary.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _, system_rows = _read_csv(system_path)
    if not system_rows:
        raise ValueError(f"效能檔案沒有資料：{system_path}")
    source = system_rows[0]
    row = {
        "run_id": str(run_directory.relative_to(report_root)),
        "generated_at": metadata.get("generated_at", ""),
        "inference_motor_id": metadata.get("inference_motor_id", ""),
        "training_motor_id": metadata.get("training_motor_id", ""),
        **{
            target: source.get(origin, "")
            for target, origin in SYSTEM_FIELD_MAP.items()
        },
    }

    performance_directory = report_root / "performance"
    history_path = performance_directory / "performance_history.csv"
    summary_csv_path = performance_directory / "performance_summary.csv"
    summary_json_path = performance_directory / "performance_summary.json"
    with _HISTORY_LOCK:
        _, history_rows = _read_csv(history_path)
        history_rows = [
            existing
            for existing in history_rows
            if existing.get("run_id") != row["run_id"]
        ]
        history_rows.append(row)
        _write_csv(history_path, HISTORY_FIELDS, history_rows)

        summaries = {
            field: descriptive_statistics(
                value
                for value in (_number(item.get(field)) for item in history_rows)
                if value is not None
            )
            for field in NUMERIC_FIELDS
        }
        summary_rows = [
            {"metric": field, **statistics}
            for field, statistics in summaries.items()
        ]
        summary_fields = [
            "metric",
            "count",
            "mean",
            "median",
            "stddev",
            "min",
            "max",
            "p90",
            "p95",
            "p99",
        ]
        _write_csv(summary_csv_path, summary_fields, summary_rows)
        summary_payload = {
            "run_count": len(history_rows),
            "metrics": summaries,
            "files": {
                "history_csv": str(history_path),
                "summary_csv": str(summary_csv_path),
                "summary_json": str(summary_json_path),
            },
        }
        temporary_json = summary_json_path.with_suffix(".json.tmp")
        temporary_json.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_json.replace(summary_json_path)
    return summary_payload


def finalize_performance_report(system_csv_path: str) -> dict:
    """Record one completed run and link its aggregate back to run metadata."""
    path = Path(system_csv_path)
    summary_path = path.parent.parent / "metadata" / "run_summary.json"
    aggregate = record_performance_run(system_csv_path)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["performance_aggregate"] = {
            "run_count": aggregate["run_count"],
            "files": aggregate["files"],
        }
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(summary_path)
    return aggregate
