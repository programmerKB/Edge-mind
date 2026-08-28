#!/usr/bin/env python3
"""Generate reproducible, research-shaped sensor CSVs for pipeline testing.

The generated values deliberately include daily ambient cycles, changing load
regimes, gradual degradation, threshold crossings, and a small cross-device
domain shift.  They are synthetic fixtures for validating the experiment
pipeline; they must not be reported as real-world model evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable


FIELDS = (
    "motor_id",
    "recorded_at",
    "temperature",
    "humidity",
    "accel_x",
    "accel_y",
    "accel_z",
    "vibration",
    "status",
    "operating_regime",
    "load_ratio",
    "wear_ratio",
    "fault_active",
    "maintenance_event",
    "is_synthetic",
)
GENERATOR_VERSION = "edgemind-research-synthetic-v2"


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Coefficients that create a controlled cross-device distribution shift."""

    motor_id: str
    ambient_offset: float
    thermal_gain: float
    cooling_rate: float
    vibration_gain: float
    phase_shift: float
    seed_offset: int


PROFILES = (
    DeviceProfile("RESEARCH-A", 0.0, 13.0, 0.075, 1.0, 0.0, 0),
    DeviceProfile("RESEARCH-B", 1.1, 14.2, 0.064, 1.18, 0.7, 10_000),
    DeviceProfile("RESEARCH-C", -1.2, 12.2, 0.083, 0.92, 1.4, 20_000),
    DeviceProfile("RESEARCH-D", 2.0, 15.1, 0.058, 1.30, 2.1, 30_000),
    DeviceProfile("RESEARCH-E", 0.6, 12.7, 0.070, 1.08, 2.8, 40_000),
    DeviceProfile("RESEARCH-F", -0.5, 14.7, 0.068, 1.22, 3.5, 50_000),
)


def _parse_start(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("start must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def _load_regime(index: int, samples_per_day: int, rng: random.Random) -> tuple[str, float]:
    """Return a repeating industrial duty cycle with deterministic variation."""
    minute_of_day = (index % samples_per_day) / samples_per_day * 24 * 60
    if minute_of_day < 6 * 60:
        name, base = "idle", 0.22
    elif minute_of_day < 8 * 60:
        name, base = "warmup", 0.48
    elif minute_of_day < 16 * 60:
        name, base = "production", 0.68
    elif minute_of_day < 20 * 60:
        name, base = "high_load", 0.90
    else:
        name, base = "cooldown", 0.38

    # Scheduled bursts expose the models to nonlinear high-temperature events.
    burst = 0.12 if index % (samples_per_day * 3) in range(180, 216) else 0.0
    return name, min(1.0, max(0.05, base + burst + rng.gauss(0.0, 0.025)))


def generate_device_rows(
    profile: DeviceProfile,
    *,
    start: datetime,
    days: int,
    interval_minutes: int,
    seed: int,
) -> list[dict]:
    """Simulate one complete, regularly sampled motor history."""
    rng = random.Random(seed + profile.seed_offset)
    samples_per_day = 24 * 60 // interval_minutes
    count = days * samples_per_day
    temperature = 25.0 + profile.ambient_offset
    rows: list[dict] = []

    for index in range(count):
        recorded_at = start + timedelta(minutes=index * interval_minutes)
        day_fraction = (index % samples_per_day) / samples_per_day
        ambient = (
            25.0
            + profile.ambient_offset
            + 2.4 * math.sin(2 * math.pi * day_fraction - 1.2 + profile.phase_shift)
            + 0.4 * math.sin(2 * math.pi * index / (samples_per_day * 7))
        )
        regime, load = _load_regime(index, samples_per_day, rng)
        # A 21-day wear cycle followed by maintenance avoids the unrealistic
        # one-way drift that made a chronological holdout a different task.
        cycle_steps = samples_per_day * 21
        cycle_position = index % cycle_steps
        wear_ratio = cycle_position / max(1, cycle_steps - 1)
        maintenance_event = cycle_position < max(1, 60 // interval_minutes)
        # Rare deterministic fault windows span several horizons and therefore
        # exercise both regression and early-warning metrics.
        fault_period = samples_per_day * (9 + profile.seed_offset // 10_000)
        fault_position = (index + profile.seed_offset // 100) % fault_period
        fault_active = samples_per_day * 0.55 <= fault_position < samples_per_day * 0.72
        if fault_active:
            load = min(1.0, load + 0.18)
        vibration_amplitude = (
            0.018 + profile.vibration_gain * (
                0.055 * load + 0.035 * wear_ratio + (0.025 if fault_active else 0.0)
            )
        )
        accel_x = vibration_amplitude * math.sin(index * 0.71) + rng.gauss(0, 0.004)
        accel_y = vibration_amplitude * math.cos(index * 0.53) + rng.gauss(0, 0.004)
        accel_z = 1.0 + 0.45 * vibration_amplitude * math.sin(index * 0.37) + rng.gauss(0, 0.002)
        vibration = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        thermal_target = (
            ambient
            + profile.thermal_gain * load
            + 3.8 * wear_ratio
            + (3.2 if fault_active else 0.0)
            + 9.0 * max(0.0, vibration - 1.0)
        )
        temperature += (
            profile.cooling_rate * (thermal_target - temperature)
            + rng.gauss(0.0, 0.055)
        )
        humidity = max(
            20.0,
            min(
                95.0,
                64.0
                - 0.62 * (temperature - ambient)
                + 5.0 * math.cos(2 * math.pi * day_fraction + 0.4)
                + rng.gauss(0.0, 0.5),
            ),
        )
        status = "synthetic-warning" if temperature >= 35.0 else "synthetic-normal"
        rows.append(
            {
                "motor_id": profile.motor_id,
                "recorded_at": recorded_at.isoformat(),
                "temperature": round(temperature, 5),
                "humidity": round(humidity, 5),
                "accel_x": round(accel_x, 7),
                "accel_y": round(accel_y, 7),
                "accel_z": round(accel_z, 7),
                "vibration": round(vibration, 7),
                "status": status,
                "operating_regime": regime,
                "load_ratio": round(load, 5),
                "wear_ratio": round(wear_ratio, 6),
                "fault_active": fault_active,
                "maintenance_event": maintenance_event,
                "is_synthetic": True,
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_manifest(count: int, gap_steps: int) -> dict:
    usable = count - 2 * gap_steps
    train_end = int(usable * 0.60)
    validation_start = train_end + gap_steps
    validation_end = validation_start + int(usable * 0.20)
    test_start = validation_end + gap_steps
    return {
        "strategy": "chronological_holdout",
        "index_unit": "complete_sequence_anchor",
        "train": {"start_index": 0, "end_index_exclusive": train_end},
        "train_validation_gap": {"steps": gap_steps},
        "validation": {
            "start_index": validation_start,
            "end_index_exclusive": min(validation_end, count),
        },
        "validation_test_gap": {"steps": gap_steps},
        "test": {"start_index": min(test_start, count), "end_index_exclusive": count},
    }


def generate_dataset(
    output_directory: Path,
    *,
    days: int = 90,
    interval_minutes: int = 5,
    seed: int = 42,
    start: datetime | None = None,
    overwrite: bool = False,
) -> dict:
    """Write both device histories and a machine-readable study manifest."""
    if days < 2:
        raise ValueError("days must be at least 2")
    if interval_minutes != 5:
        raise ValueError("this research protocol requires interval_minutes=5")
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start must include an explicit timezone offset")
    start = start.astimezone(timezone.utc)
    expected_seconds = interval_minutes * 60
    if start.microsecond or int(start.timestamp()) % expected_seconds:
        raise ValueError("start must be on the UTC 5-minute grid")
    output_directory = output_directory.resolve()
    output_paths = [
        *(output_directory / f"{profile.motor_id}.csv" for profile in PROFILES),
        output_directory / "manifest.json",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "dataset release already exists; choose a new output directory or "
            "pass --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    gap_steps = 30 // interval_minutes
    files: dict[str, str] = {}
    file_sha256: dict[str, str] = {}
    counts: dict[str, int] = {}
    sequence_counts: dict[str, int] = {}
    for profile in PROFILES:
        rows = generate_device_rows(
            profile,
            start=start,
            days=days,
            interval_minutes=interval_minutes,
            seed=seed,
        )
        path = output_directory / f"{profile.motor_id}.csv"
        _write_csv(path, rows)
        files[profile.motor_id] = path.name
        file_sha256[profile.motor_id] = _sha256(path)
        counts[profile.motor_id] = len(rows)
        sequence_counts[profile.motor_id] = max(
            0,
            len(rows) - (12 - 1) - gap_steps,
        )

    generator = {
        "version": GENERATOR_VERSION,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "days": days,
        "interval_minutes": interval_minutes,
        "seed": seed,
        "start": start.isoformat(),
        "profiles": [asdict(profile) for profile in PROFILES],
    }
    sequence_contract = {
        "features": ["temperature", "humidity", "accel_x", "accel_y", "accel_z"],
        "history_steps": 12,
        "history_minutes": 60,
        "horizons_minutes": [5, 10, 15, 20, 25, 30],
        "gap_steps": gap_steps,
    }
    stable_identity = {
        "generator": generator,
        "sequence_contract": sequence_contract,
        "file_sha256": file_sha256,
    }
    dataset_content_sha256 = hashlib.sha256(
        json.dumps(
            stable_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "2.0",
        "protocol_version": "temperature-trajectory-v2",
        "dataset_kind": "synthetic_pipeline_validation_only",
        "research_claims_allowed": False,
        "warning": (
            "Synthetic data may validate software behavior but must not be used "
            "as evidence of real equipment forecasting performance."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": generator,
        "sequence_contract": sequence_contract,
        "recommended_evaluation": {
            "within_device": _split_manifest(
                sequence_counts[PROFILES[0].motor_id], gap_steps
            ),
            "cross_device": {
                "fit_device": PROFILES[0].motor_id,
                "reserved_external_test_device": PROFILES[1].motor_id,
                "fit_on_external_device": False,
                "additional_robustness_devices": [
                    profile.motor_id for profile in PROFILES[2:]
                ],
            },
            "walk_forward_folds": 3,
        },
        "files": files,
        "file_sha256": file_sha256,
        "dataset_content_sha256": dataset_content_sha256,
        "row_counts": counts,
        "expected_complete_sequence_counts": sequence_counts,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "research_datasets",
    )
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2026-01-01T00:00:00+00:00")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing synthetic dataset release",
    )
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output_dir.resolve(),
        days=args.days,
        interval_minutes=args.interval_minutes,
        seed=args.seed,
        start=_parse_start(args.start),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
