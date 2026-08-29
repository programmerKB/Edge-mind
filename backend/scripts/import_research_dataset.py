#!/usr/bin/env python3
"""Validate and import EdgeMind research CSVs into the configured database.

By default an existing ``(motor_id, recorded_at)`` pair is skipped, making the
command safe to repeat.  ``--replace-device`` is explicit because it deletes
only the device IDs present in the input files before importing them again.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys


# Allow running this file directly from either the repository root or backend.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

REQUIRED_FIELDS = (
    "motor_id",
    "recorded_at",
    "temperature",
    "humidity",
    "accel_x",
    "accel_y",
    "accel_z",
)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def _finite(row: dict[str, str], name: str, line: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"line {line}: {name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"line {line}: {name} must be finite")
    return value


def read_and_validate(paths: list[Path], expected_interval_minutes: int) -> list[dict]:
    """Return normalized rows after schema, value, duplicate, and cadence checks."""
    if expected_interval_minutes <= 0:
        raise ValueError("expected_interval_minutes must be positive")
    normalized: list[dict] = []
    seen: set[tuple[str, datetime]] = set()
    timestamps: dict[str, list[datetime]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
            for line, row in enumerate(reader, start=2):
                motor_id = (row.get("motor_id") or "").strip()
                if not motor_id:
                    raise ValueError(f"{path}:{line}: motor_id is empty")
                try:
                    recorded_at = _timestamp(row.get("recorded_at") or "")
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line}: recorded_at must be an ISO-8601 timestamp"
                    ) from error
                expected_seconds = expected_interval_minutes * 60
                if (
                    recorded_at.microsecond
                    or int(recorded_at.timestamp()) % expected_seconds
                ):
                    raise ValueError(
                        f"{path}:{line}: recorded_at is off the UTC "
                        f"{expected_interval_minutes}-minute grid"
                    )
                key = (motor_id, recorded_at)
                if key in seen:
                    raise ValueError(f"{path}:{line}: duplicate motor_id/timestamp pair")
                seen.add(key)
                temperature = _finite(row, "temperature", line)
                humidity = _finite(row, "humidity", line)
                accel_x = _finite(row, "accel_x", line)
                accel_y = _finite(row, "accel_y", line)
                accel_z = _finite(row, "accel_z", line)
                if not 0 <= humidity <= 100:
                    raise ValueError(f"{path}:{line}: humidity must be between 0 and 100")
                vibration_text = row.get("vibration")
                vibration = (
                    _finite(row, "vibration", line)
                    if vibration_text not in (None, "")
                    else math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                )
                status = (row.get("status") or "").strip() or "research-import"
                is_synthetic = str(row.get("is_synthetic") or "").strip().lower() in {
                    "1", "true", "yes", "y",
                }
                if is_synthetic and not status.lower().startswith(("demo", "synthetic")):
                    status = f"synthetic-{status}"
                normalized.append(
                    {
                        "motor_id": motor_id,
                        "temperature": temperature,
                        "humidity": humidity,
                        "accel_x": accel_x,
                        "accel_y": accel_y,
                        "accel_z": accel_z,
                        "vibration": vibration,
                        "status": status[:50],
                        "recorded_at": recorded_at,
                    }
                )
                timestamps.setdefault(motor_id, []).append(recorded_at)

    expected_seconds = expected_interval_minutes * 60
    for motor_id, values in timestamps.items():
        ordered = sorted(values)
        bad_gaps = [
            (right - left).total_seconds()
            for left, right in zip(ordered, ordered[1:])
            if (right - left).total_seconds() != expected_seconds
        ]
        if bad_gaps:
            raise ValueError(
                f"{motor_id}: irregular cadence; expected exactly "
                f"{expected_interval_minutes} minutes (bad gaps={len(bad_gaps)})"
            )
    return sorted(normalized, key=lambda item: (item["motor_id"], item["recorded_at"]))


def import_rows(rows: list[dict], database_url: str, replace_device: bool) -> dict:
    """Insert validated rows transactionally and report imported/skipped counts."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from edgemind.infrastructure.persistence.database import Base
    from edgemind.infrastructure.persistence.models import MotorSensorData

    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    imported = 0
    skipped = 0
    devices = sorted({row["motor_id"] for row in rows})
    try:
        if replace_device:
            session.query(MotorSensorData).filter(
                MotorSensorData.motor_id.in_(devices)
            ).delete(synchronize_session=False)
        existing = {
            (motor_id, recorded_at)
            for motor_id, recorded_at in session.query(
                MotorSensorData.motor_id,
                MotorSensorData.recorded_at,
            ).filter(MotorSensorData.motor_id.in_(devices))
        }
        for row in rows:
            key = (row["motor_id"], row["recorded_at"])
            # SQLite may return naive datetimes; compare their ISO wall time too.
            duplicate = key in existing or any(
                device == key[0]
                and timestamp.replace(tzinfo=timezone.utc) == key[1]
                for device, timestamp in existing
                if timestamp is not None and timestamp.tzinfo is None
            )
            if duplicate:
                skipped += 1
                continue
            session.add(MotorSensorData(**row))
            existing.add(key)
            imported += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()
    return {"devices": devices, "imported": imported, "skipped": skipped}


def main() -> None:
    from edgemind.infrastructure.config import settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--expected-interval-minutes", type=int, default=5)
    parser.add_argument("--replace-device", action="store_true")
    args = parser.parse_args()
    paths = [path.resolve() for path in args.csv]
    rows = read_and_validate(paths, args.expected_interval_minutes)
    result = import_rows(rows, args.database_url, args.replace_device)
    print(
        f"Validated {len(rows)} rows; imported={result['imported']}, "
        f"skipped={result['skipped']}, devices={','.join(result['devices'])}"
    )


if __name__ == "__main__":
    main()
