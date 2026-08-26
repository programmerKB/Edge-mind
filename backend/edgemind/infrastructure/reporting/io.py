"""CSV serialization and source-dataset exports for reports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from edgemind.domain.demo_data import (
    DEMO_INFERENCE_MOTOR_ID,
    DEMO_TRAINING_MOTOR_ID,
)


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


def aware_datetime(value: datetime) -> datetime:
    """Attach UTC to legacy naive timestamps without shifting their value."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def iso_datetime(value: datetime | None) -> str:
    """Serialize a nullable timestamp consistently for CSV and JSON files."""
    return aware_datetime(value).isoformat() if value else ""


def is_finite(value: Any) -> bool:
    """Return whether a value can be represented as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rounded(value: float | None, digits: int = 6) -> float | str:
    """Round finite metrics and use an empty CSV cell for missing values."""
    return round(value, digits) if value is not None and is_finite(value) else ""


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    """Write an Excel-compatible UTF-8 CSV, creating parent folders first."""
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
            row[field] = iso_datetime(value) if field == "recorded_at" else value
        rows.append(row)
    write_csv(path, SENSOR_FIELDS, rows)


def export_demo_datasets(
    training_records: Iterable[Any],
    inference_records: Iterable[Any],
    report_root: Path,
) -> dict[str, str]:
    """Refresh categorized demo CSVs from domain records supplied by a caller."""
    destinations = {
        DEMO_TRAINING_MOTOR_ID: report_root
        / "datasets"
        / "training"
        / f"{DEMO_TRAINING_MOTOR_ID}.csv",
        DEMO_INFERENCE_MOTOR_ID: report_root
        / "datasets"
        / "inference"
        / f"{DEMO_INFERENCE_MOTOR_ID}.csv",
    }
    export_sensor_dataset(
        training_records,
        destinations[DEMO_TRAINING_MOTOR_ID],
    )
    export_sensor_dataset(
        inference_records,
        destinations[DEMO_INFERENCE_MOTOR_ID],
    )
    return {key: str(value) for key, value in destinations.items()}
