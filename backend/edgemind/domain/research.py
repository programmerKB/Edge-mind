"""Framework-independent multi-horizon forecasting research engine.

The production forecast endpoint intentionally remains small and backwards
compatible.  This module contains the stricter experimental pipeline used to
compare sequence models fairly:

* exact five-minute alignment (no nearest-future target substitution),
* a 12-reading history window and configurable direct horizons,
* chronological train/validation/test and expanding walk-forward splits with
  an embargo at least as long as the furthest forecast horizon,
* executable dependency-free baselines, and
* trajectory, risk, accuracy, latency, and serialized-size measurements.

The module imports only the Python standard library and other domain
calculations.  Optional ML implementations can be registered from an outward
infrastructure layer without making the domain depend on those packages.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from statistics import fmean, pstdev
import time
from typing import Any

from edgemind.domain.evaluation import (
    binary_score_metrics,
    classification_metrics,
    descriptive_statistics,
    regression_metrics,
)
from edgemind.domain.formal_analysis import (
    TemperatureObservation,
    WarningSignal,
    evaluate_event_warnings,
    paired_block_bootstrap,
)


BASE_FEATURE_NAMES = (
    "temperature",
    "humidity",
    "accel_x",
    "accel_y",
    "accel_z",
)
DEFAULT_HORIZONS_MINUTES = (5, 10, 15, 20, 25, 30)
MAX_HORIZON_MINUTES = 60
DEFAULT_MODEL_NAMES = (
    "ridge_direct",
    "ridge_history_trend",
    "dlinear",
    "lstm",
    "tcn",
    "patchtst",
)
DEFAULT_RIDGE_ALPHA_CANDIDATES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
OPTIONAL_MODEL_NAMES = (
    "dlinear",
    "lstm",
    "tcn",
    "patchtst",
)
UNKNOWN_DEVICE_ID = "UNKNOWN"


class ResearchError(ValueError):
    """Raised when an experimental data contract cannot be satisfied."""


def _as_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _as_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        try:
            value = datetime.fromisoformat(candidate)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _sample_id(example: "SequenceExample", config: "ResearchConfig") -> str:
    """Return a stable window identity shared by every model and horizon row."""

    identity = {
        "protocol": "temperature-trajectory-v1",
        "device_id": example.device_id,
        "origin_time": _iso(example.anchor_time),
        "sampling_minutes": config.sampling_minutes,
        "history_steps": config.history_steps,
        "horizons_minutes": list(config.horizons_minutes),
        "feature_names": list(BASE_FEATURE_NAMES),
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """One reproducible experimental contract.

    ``history_steps=12`` means twelve five-minute observations (a nominal
    60-minute input window, spanning timestamps t-55 through t).  Horizons are
    expressed in minutes so extending an experiment through +60 only requires
    passing ``tuple(range(5, 61, 5))``.
    """

    sampling_minutes: int = 5
    history_steps: int = 12
    horizons_minutes: tuple[int, ...] = DEFAULT_HORIZONS_MINUTES
    threshold_c: float = 35.0
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    gap_steps: int | None = None
    ridge_alpha: float = 0.1
    ridge_alpha_candidates: tuple[float, ...] = DEFAULT_RIDGE_ALPHA_CANDIDATES
    max_training_epochs: int = 160
    early_stopping_patience: int = 16
    training_batch_size: int = 128
    medium_risk_margin_c: float = 2.0
    rapid_heating_c_per_minute: float = 0.10
    walk_forward_folds: int = 3
    event_merge_gap_minutes: int = 15
    warning_lookback_minutes: int = 30
    warning_cooldown_minutes: int = 5
    bootstrap_repetitions: int = 10_000
    bootstrap_seed: int = 17
    random_seed: int = 42

    def __post_init__(self) -> None:
        horizons = tuple(int(value) for value in self.horizons_minutes)
        object.__setattr__(self, "horizons_minutes", horizons)
        if self.sampling_minutes <= 0:
            raise ResearchError("sampling_minutes must be positive")
        if self.history_steps < 2:
            raise ResearchError("history_steps must contain at least two readings")
        if not horizons or any(value <= 0 for value in horizons):
            raise ResearchError("horizons_minutes must contain positive values")
        if any(value > MAX_HORIZON_MINUTES for value in horizons):
            raise ResearchError(
                f"horizons_minutes cannot exceed {MAX_HORIZON_MINUTES} minutes"
            )
        if tuple(sorted(set(horizons))) != horizons:
            raise ResearchError("horizons_minutes must be unique and increasing")
        if any(value % self.sampling_minutes for value in horizons):
            raise ResearchError(
                "every horizon must be an exact multiple of sampling_minutes"
            )
        if not math.isfinite(float(self.threshold_c)):
            raise ResearchError("threshold_c must be finite")
        if not 0 < self.train_fraction < 1:
            raise ResearchError("train_fraction must be between zero and one")
        if not 0 < self.validation_fraction < 1:
            raise ResearchError("validation_fraction must be between zero and one")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ResearchError("train and validation fractions must leave a test set")
        if self.ridge_alpha <= 0 or not math.isfinite(float(self.ridge_alpha)):
            raise ResearchError("ridge_alpha must be finite and positive")
        candidates = tuple(float(value) for value in self.ridge_alpha_candidates)
        object.__setattr__(self, "ridge_alpha_candidates", candidates)
        if not candidates or any(value <= 0 or not math.isfinite(value) for value in candidates):
            raise ResearchError("ridge_alpha_candidates must be finite and positive")
        if tuple(sorted(set(candidates))) != candidates:
            raise ResearchError("ridge_alpha_candidates must be unique and increasing")
        if self.max_training_epochs < 1:
            raise ResearchError("max_training_epochs must be positive")
        if not 1 <= self.early_stopping_patience <= self.max_training_epochs:
            raise ResearchError(
                "early_stopping_patience must be between one and max_training_epochs"
            )
        if self.training_batch_size < 1:
            raise ResearchError("training_batch_size must be positive")
        if self.medium_risk_margin_c < 0:
            raise ResearchError("medium_risk_margin_c cannot be negative")
        if self.rapid_heating_c_per_minute < 0:
            raise ResearchError("rapid_heating_c_per_minute cannot be negative")
        if self.walk_forward_folds < 1:
            raise ResearchError("walk_forward_folds must be positive")
        for name, value in (
            ("event_merge_gap_minutes", self.event_merge_gap_minutes),
            ("warning_lookback_minutes", self.warning_lookback_minutes),
            ("warning_cooldown_minutes", self.warning_cooldown_minutes),
        ):
            if value < 0 or value % self.sampling_minutes:
                raise ResearchError(
                    f"{name} must be a non-negative multiple of sampling_minutes"
                )
        if self.warning_lookback_minutes <= 0:
            raise ResearchError("warning_lookback_minutes must be positive")
        if self.bootstrap_repetitions < 1:
            raise ResearchError("bootstrap_repetitions must be positive")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ResearchError("random_seed must be an integer")
        if self.gap_steps is not None:
            if self.gap_steps < self.minimum_safe_gap_steps:
                raise ResearchError(
                    "gap_steps must be at least the furthest horizon in samples "
                    f"({self.minimum_safe_gap_steps})"
                )

    @property
    def history_minutes(self) -> int:
        """Nominal duration represented by the configured reading count."""
        return self.history_steps * self.sampling_minutes

    @property
    def minimum_safe_gap_steps(self) -> int:
        return max(self.horizons_minutes) // self.sampling_minutes

    @property
    def effective_gap_steps(self) -> int:
        return (
            self.minimum_safe_gap_steps
            if self.gap_steps is None
            else self.gap_steps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampling_minutes": self.sampling_minutes,
            "history_steps": self.history_steps,
            "history_minutes": self.history_minutes,
            "horizons_minutes": list(self.horizons_minutes),
            "threshold_c": float(self.threshold_c),
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": 1 - self.train_fraction - self.validation_fraction,
            "gap_steps": self.effective_gap_steps,
            "gap_minutes": self.effective_gap_steps * self.sampling_minutes,
            "ridge_alpha": self.ridge_alpha,
            "ridge_alpha_candidates": list(self.ridge_alpha_candidates),
            "max_training_epochs": self.max_training_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "training_batch_size": self.training_batch_size,
            "medium_risk_margin_c": self.medium_risk_margin_c,
            "rapid_heating_c_per_minute": self.rapid_heating_c_per_minute,
            "walk_forward_folds": self.walk_forward_folds,
            "event_merge_gap_minutes": self.event_merge_gap_minutes,
            "warning_lookback_minutes": self.warning_lookback_minutes,
            "warning_cooldown_minutes": self.warning_cooldown_minutes,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True, slots=True)
class _AlignedPoint:
    device_id: str
    recorded_at: datetime
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SequenceExample:
    """A complete history matrix paired with one future trajectory."""

    device_id: str
    anchor_time: datetime
    history_times: tuple[datetime, ...]
    history: tuple[tuple[float, ...], ...]
    target_times: tuple[datetime, ...]
    targets: tuple[float, ...]

    @property
    def current_temperature(self) -> float:
        return float(self.history[-1][0])

    def to_dict(self, *, include_history: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "device_id": self.device_id,
            "anchor_time": _iso(self.anchor_time),
            "target_times": [_iso(value) for value in self.target_times],
            "targets": list(self.targets),
            "history_shape": [len(self.history), len(self.history[0])],
        }
        if include_history:
            payload["history_times"] = [_iso(value) for value in self.history_times]
            payload["history"] = [list(row) for row in self.history]
        return payload


@dataclass(frozen=True, slots=True)
class InferenceSequence:
    """Latest exact history window whose future truth is not available yet."""

    device_id: str
    anchor_time: datetime
    history_times: tuple[datetime, ...]
    history: tuple[tuple[float, ...], ...]
    target_times: tuple[datetime, ...]

    @property
    def current_temperature(self) -> float:
        return float(self.history[-1][0])

    def to_dict(self, *, include_history: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "device_id": self.device_id,
            "anchor_time": _iso(self.anchor_time),
            "current_temperature_c": self.current_temperature,
            "history_shape": [len(self.history), len(self.history[0])],
            "history_start_time": _iso(self.history_times[0]),
            "history_end_time": _iso(self.history_times[-1]),
            "target_times": [_iso(value) for value in self.target_times],
            "truth_status": "pending",
        }
        if include_history:
            payload["history_times"] = [_iso(value) for value in self.history_times]
            payload["history"] = [list(row) for row in self.history]
        return payload


@dataclass(frozen=True, slots=True)
class AlignmentQuality:
    input_rows: int
    filtered_out_rows: int
    accepted_rows: int
    invalid_timestamp_rows: int
    incomplete_feature_rows: int
    nonfinite_feature_rows: int
    duplicate_timestamp_count: int
    duplicate_rows_excluded: int
    out_of_order_rows: int
    irregular_interval_count: int
    missing_interval_steps: int
    candidate_anchor_count: int
    complete_sequence_count: int
    dropped_unaligned_anchors: int

    def to_dict(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    config: ResearchConfig
    examples: tuple[SequenceExample, ...]
    quality: AlignmentQuality
    device_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        anchors = [example.anchor_time for example in self.examples]
        return {
            "feature_names": list(BASE_FEATURE_NAMES),
            "history_shape": [self.config.history_steps, len(BASE_FEATURE_NAMES)],
            "horizons_minutes": list(self.config.horizons_minutes),
            "device_ids": list(self.device_ids),
            "sequence_count": len(self.examples),
            "first_anchor_time": _iso(min(anchors)) if anchors else None,
            "last_anchor_time": _iso(max(anchors)) if anchors else None,
            "quality": self.quality.to_dict(),
        }


def _feature_tuple(record: Any) -> tuple[float, ...] | str:
    raw = tuple(_as_value(record, name) for name in BASE_FEATURE_NAMES)
    if any(value is None for value in raw):
        return "incomplete"
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return "incomplete"
    if not all(math.isfinite(value) for value in values):
        return "nonfinite"
    return values


def build_sequence_dataset(
    records: Iterable[Any],
    config: ResearchConfig | None = None,
    *,
    device_id: str | None = None,
) -> SequenceDataset:
    """Validate, align, and transform readings into complete sequences.

    Targets must exist at their exact relative timestamps.  Missing or
    duplicated readings invalidate affected anchors instead of being silently
    interpolated or paired to a nearby future observation.
    """

    config = config or ResearchConfig()
    rows = list(records)
    filtered_out_rows = 0
    invalid_timestamp_rows = 0
    incomplete_feature_rows = 0
    nonfinite_feature_rows = 0
    out_of_order_rows = 0
    prior_input_time: dict[str, datetime] = {}
    valid: list[_AlignedPoint] = []

    for record in rows:
        raw_device = _as_value(record, "motor_id", UNKNOWN_DEVICE_ID)
        current_device = str(raw_device or UNKNOWN_DEVICE_ID)
        if device_id is not None and current_device != str(device_id):
            filtered_out_rows += 1
            continue
        recorded_at = _as_utc_datetime(_as_value(record, "recorded_at"))
        if recorded_at is None:
            invalid_timestamp_rows += 1
            continue
        previous = prior_input_time.get(current_device)
        if previous is not None and recorded_at < previous:
            out_of_order_rows += 1
        prior_input_time[current_device] = recorded_at
        features = _feature_tuple(record)
        if features == "incomplete":
            incomplete_feature_rows += 1
            continue
        if features == "nonfinite":
            nonfinite_feature_rows += 1
            continue
        valid.append(_AlignedPoint(current_device, recorded_at, features))

    grouped: dict[tuple[str, datetime], list[_AlignedPoint]] = {}
    for point in valid:
        grouped.setdefault((point.device_id, point.recorded_at), []).append(point)
    duplicate_groups = {
        key: points for key, points in grouped.items() if len(points) > 1
    }
    unique_points = [
        points[0] for key, points in grouped.items() if key not in duplicate_groups
    ]
    unique_points.sort(key=lambda point: (point.device_id, point.recorded_at))

    interval = timedelta(minutes=config.sampling_minutes)
    irregular_interval_count = 0
    missing_interval_steps = 0
    points_by_device: dict[str, list[_AlignedPoint]] = {}
    for point in unique_points:
        points_by_device.setdefault(point.device_id, []).append(point)
    for points in points_by_device.values():
        for previous, current in zip(points, points[1:]):
            delta = current.recorded_at - previous.recorded_at
            if delta != interval:
                irregular_interval_count += 1
                delta_seconds = delta.total_seconds()
                interval_seconds = interval.total_seconds()
                if delta_seconds > interval_seconds:
                    missing_interval_steps += max(
                        0,
                        int(delta_seconds // interval_seconds) - 1,
                    )

    examples: list[SequenceExample] = []
    for current_device, points in points_by_device.items():
        by_time = {point.recorded_at: point for point in points}
        for anchor in points:
            history_times = tuple(
                anchor.recorded_at
                - interval * (config.history_steps - 1 - index)
                for index in range(config.history_steps)
            )
            target_times = tuple(
                anchor.recorded_at + timedelta(minutes=horizon)
                for horizon in config.horizons_minutes
            )
            if not all(value in by_time for value in history_times):
                continue
            if not all(value in by_time for value in target_times):
                continue
            examples.append(
                SequenceExample(
                    device_id=current_device,
                    anchor_time=anchor.recorded_at,
                    history_times=history_times,
                    history=tuple(by_time[value].values for value in history_times),
                    target_times=target_times,
                    targets=tuple(by_time[value].values[0] for value in target_times),
                )
            )
    examples.sort(key=lambda item: (item.anchor_time, item.device_id))
    accepted_rows = len(unique_points)
    quality = AlignmentQuality(
        input_rows=len(rows),
        filtered_out_rows=filtered_out_rows,
        accepted_rows=accepted_rows,
        invalid_timestamp_rows=invalid_timestamp_rows,
        incomplete_feature_rows=incomplete_feature_rows,
        nonfinite_feature_rows=nonfinite_feature_rows,
        duplicate_timestamp_count=len(duplicate_groups),
        duplicate_rows_excluded=sum(len(points) for points in duplicate_groups.values()),
        out_of_order_rows=out_of_order_rows,
        irregular_interval_count=irregular_interval_count,
        missing_interval_steps=missing_interval_steps,
        candidate_anchor_count=accepted_rows,
        complete_sequence_count=len(examples),
        dropped_unaligned_anchors=accepted_rows - len(examples),
    )
    return SequenceDataset(
        config=config,
        examples=tuple(examples),
        quality=quality,
        device_ids=tuple(sorted(points_by_device)),
    )


def build_inference_sequence(
    records: Iterable[Any],
    config: ResearchConfig | None = None,
    *,
    device_id: str | None = None,
) -> InferenceSequence:
    """Build the newest exact history window without requiring future labels.

    The newest timestamp is the forecast origin.  Every one of the configured
    history grid points must exist exactly once and contain all five finite
    features; the function never forward-fills, interpolates, or silently uses
    an older origin.
    """

    config = config or ResearchConfig()
    rows = list(records)
    if device_id is not None:
        rows = [
            record
            for record in rows
            if str(_as_value(record, "motor_id", UNKNOWN_DEVICE_ID)) == str(device_id)
        ]
    if not rows:
        raise ResearchError("live inference requires sensor readings")
    devices = {
        str(_as_value(record, "motor_id", UNKNOWN_DEVICE_ID) or UNKNOWN_DEVICE_ID)
        for record in rows
    }
    if len(devices) != 1:
        raise ResearchError("live inference requires exactly one device")
    current_device = next(iter(devices))

    grouped: dict[datetime, list[Any]] = {}
    invalid_timestamp_count = 0
    for record in rows:
        recorded_at = _as_utc_datetime(_as_value(record, "recorded_at"))
        if recorded_at is None:
            invalid_timestamp_count += 1
            continue
        grouped.setdefault(recorded_at, []).append(record)
    if not grouped:
        raise ResearchError("live inference has no valid timestamps")
    anchor_time = max(grouped)
    interval = timedelta(minutes=config.sampling_minutes)
    history_times = tuple(
        anchor_time - interval * (config.history_steps - 1 - index)
        for index in range(config.history_steps)
    )
    missing = [value for value in history_times if value not in grouped]
    if missing:
        raise ResearchError(
            "latest live window is incomplete: "
            f"missing {len(missing)} of {config.history_steps} exact grid readings"
        )
    duplicates = [value for value in history_times if len(grouped[value]) != 1]
    if duplicates:
        raise ResearchError(
            "latest live window contains duplicated timestamps; resolve them before inference"
        )
    history: list[tuple[float, ...]] = []
    for timestamp in history_times:
        features = _feature_tuple(grouped[timestamp][0])
        if features == "incomplete":
            raise ResearchError(
                f"live reading at {_iso(timestamp)} has incomplete features"
            )
        if features == "nonfinite":
            raise ResearchError(
                f"live reading at {_iso(timestamp)} has non-finite features"
            )
        history.append(features)
    target_times = tuple(
        anchor_time + timedelta(minutes=horizon)
        for horizon in config.horizons_minutes
    )
    sequence = InferenceSequence(
        device_id=current_device,
        anchor_time=anchor_time,
        history_times=history_times,
        history=tuple(history),
        target_times=target_times,
    )
    # Invalid legacy rows outside the latest exact window are deliberately not
    # fatal, but keeping this branch documents that they were considered.
    _ = invalid_timestamp_count
    return sequence


def _partition_summary(examples: Sequence[SequenceExample]) -> dict[str, Any]:
    return {
        "sample_count": len(examples),
        "first_anchor_time": _iso(examples[0].anchor_time) if examples else None,
        "last_anchor_time": _iso(examples[-1].anchor_time) if examples else None,
        "last_target_time": (
            _iso(max(examples[-1].target_times)) if examples else None
        ),
    }


def _has_safe_boundary(
    earlier: Sequence[SequenceExample],
    later: Sequence[SequenceExample],
) -> bool:
    if not earlier or not later:
        return False
    latest_label = max(time for item in earlier for time in item.target_times)
    first_later_anchor = min(item.anchor_time for item in later)
    return latest_label < first_later_anchor


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: tuple[SequenceExample, ...]
    validation: tuple[SequenceExample, ...]
    test: tuple[SequenceExample, ...]
    gap_steps: int
    skipped_sequence_count: int

    def to_dict(self, sampling_minutes: int = 5) -> dict[str, Any]:
        return {
            "strategy": "chronological_train_validation_test",
            "train": _partition_summary(self.train),
            "validation": _partition_summary(self.validation),
            "test": _partition_summary(self.test),
            "gap_steps": self.gap_steps,
            "gap_minutes": self.gap_steps * sampling_minutes,
            "skipped_sequence_count": self.skipped_sequence_count,
            "leakage_audit": {
                "train_targets_end_before_validation_anchor": _has_safe_boundary(
                    self.train, self.validation
                ),
                "validation_targets_end_before_test_anchor": _has_safe_boundary(
                    self.validation, self.test
                ),
            },
        }


def _require_single_device(dataset: SequenceDataset) -> None:
    if len(dataset.device_ids) != 1:
        raise ResearchError(
            "chronological research splits require exactly one device; "
            "build one training dataset per device"
        )


def chronological_split(
    dataset: SequenceDataset,
    config: ResearchConfig | None = None,
) -> DatasetSplit:
    """Create ordered train/validation/test partitions with two embargoes."""

    config = config or dataset.config
    _require_single_device(dataset)
    examples = dataset.examples
    gap = config.effective_gap_steps
    usable = len(examples) - 2 * gap
    if usable < 3:
        raise ResearchError(
            "not enough complete sequences for train/validation/test after "
            f"two {gap}-step leakage gaps"
        )
    train_count = max(1, int(usable * config.train_fraction))
    validation_count = max(1, int(usable * config.validation_fraction))
    if train_count + validation_count >= usable:
        validation_count = 1
        train_count = usable - 2
    test_count = usable - train_count - validation_count

    validation_start = train_count + gap
    test_start = validation_start + validation_count + gap
    split = DatasetSplit(
        train=tuple(examples[:train_count]),
        validation=tuple(
            examples[validation_start : validation_start + validation_count]
        ),
        test=tuple(examples[test_start : test_start + test_count]),
        gap_steps=gap,
        skipped_sequence_count=2 * gap,
    )
    if not _has_safe_boundary(split.train, split.validation):
        raise ResearchError("unsafe train/validation target overlap detected")
    if not _has_safe_boundary(split.validation, split.test):
        raise ResearchError("unsafe validation/test target overlap detected")
    return split


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    index: int
    train: tuple[SequenceExample, ...]
    test: tuple[SequenceExample, ...]
    gap_steps: int

    def to_dict(self, sampling_minutes: int = 5) -> dict[str, Any]:
        return {
            "fold": self.index,
            "train": _partition_summary(self.train),
            "test": _partition_summary(self.test),
            "gap_steps": self.gap_steps,
            "gap_minutes": self.gap_steps * sampling_minutes,
            "leakage_audit": {
                "train_targets_end_before_test_anchor": _has_safe_boundary(
                    self.train, self.test
                )
            },
        }


def walk_forward_splits(
    dataset: SequenceDataset,
    config: ResearchConfig | None = None,
    *,
    examples: Sequence[SequenceExample] | None = None,
    n_splits: int | None = None,
    test_size: int | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Return expanding-window folds with a horizon-length gap per fold.

    ``examples`` can constrain cross-validation to an outer split's
    development period.  The experiment runner uses this boundary so the
    locked test partition and its preceding embargo can never enter model
    selection folds.
    """

    config = config or dataset.config
    _require_single_device(dataset)
    scoped_examples = tuple(dataset.examples if examples is None else examples)
    if not scoped_examples:
        raise ResearchError("walk-forward requires at least one sequence")
    if len({example.device_id for example in scoped_examples}) != 1:
        raise ResearchError("walk-forward examples must belong to one device")
    if tuple(sorted(scoped_examples, key=lambda item: item.anchor_time)) != scoped_examples:
        raise ResearchError("walk-forward examples must be chronological")
    split_count = config.walk_forward_folds if n_splits is None else n_splits
    if split_count < 1:
        raise ResearchError("n_splits must be positive")
    gap = config.effective_gap_steps
    total = len(scoped_examples)
    if test_size is None:
        test_size = max(1, (total - gap) // (split_count + 2))
    if test_size < 1:
        raise ResearchError("test_size must be positive")
    initial_train_count = total - gap - split_count * test_size
    if initial_train_count < 1:
        raise ResearchError("not enough sequences for the requested walk-forward folds")

    folds: list[WalkForwardFold] = []
    for offset in range(split_count):
        train_end = initial_train_count + offset * test_size
        test_start = train_end + gap
        fold = WalkForwardFold(
            index=offset + 1,
            train=tuple(scoped_examples[:train_end]),
            test=tuple(scoped_examples[test_start : test_start + test_size]),
            gap_steps=gap,
        )
        if not _has_safe_boundary(fold.train, fold.test):
            raise ResearchError(f"unsafe target overlap detected in fold {fold.index}")
        folds.append(fold)
    return tuple(folds)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]], values: Sequence[float]
) -> list[float]:
    size = len(values)
    augmented = [list(matrix[row]) + [float(values[row])] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ResearchError("ridge parameter matrix could not be solved")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][index] - factor * augmented[column][index]
                    for index in range(size + 1)
                ]
    return [augmented[row][-1] for row in range(size)]


def _fit_multioutput_ridge(
    rows: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    alpha: float,
) -> dict[str, Any]:
    if not rows or len(rows) != len(targets):
        raise ResearchError("ridge training requires paired non-empty rows and targets")
    width = len(rows[0])
    target_width = len(targets[0])
    if width < 1 or target_width < 1:
        raise ResearchError("ridge rows and targets cannot be empty")
    if any(len(row) != width for row in rows):
        raise ResearchError("ridge feature rows must have equal width")
    if any(len(row) != target_width for row in targets):
        raise ResearchError("ridge target rows must have equal width")

    means = [fmean(float(row[index]) for row in rows) for index in range(width)]
    scales: list[float] = []
    for index in range(width):
        variance = fmean(
            (float(row[index]) - means[index]) ** 2 for row in rows
        )
        scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    normalized = [
        [1.0]
        + [
            (float(row[index]) - means[index]) / scales[index]
            for index in range(width)
        ]
        for row in rows
    ]
    augmented_width = width + 1
    gram = [[0.0] * augmented_width for _ in range(augmented_width)]
    for row in normalized:
        for left in range(augmented_width):
            left_value = row[left]
            for right in range(left, augmented_width):
                gram[left][right] += left_value * row[right]
    for left in range(augmented_width):
        for right in range(left):
            gram[left][right] = gram[right][left]
    for index in range(1, augmented_width):
        gram[index][index] += alpha

    coefficients: list[list[float]] = []
    for output in range(target_width):
        rhs = [
            sum(
                row[index] * float(target[output])
                for row, target in zip(normalized, targets)
            )
            for index in range(augmented_width)
        ]
        coefficients.append(_solve_linear_system(gram, rhs))
    return {
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
    }


def _predict_multioutput_ridge(
    state: Mapping[str, Any], row: Sequence[float]
) -> tuple[float, ...]:
    means = state["means"]
    scales = state["scales"]
    if len(row) != len(means):
        raise ResearchError("ridge inference feature width does not match training")
    normalized = [1.0] + [
        (float(value) - float(mean)) / float(scale)
        for value, mean, scale in zip(row, means, scales)
    ]
    predictions = tuple(
        sum(float(weight) * value for weight, value in zip(output, normalized))
        for output in state["coefficients"]
    )
    if not all(math.isfinite(value) for value in predictions):
        raise ResearchError("model produced a non-finite prediction")
    return predictions


def _feature_indexes(feature_names: Sequence[str]) -> tuple[int, ...]:
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ResearchError("feature_names must be non-empty and unique")
    unknown = [name for name in names if name not in BASE_FEATURE_NAMES]
    if unknown:
        raise ResearchError(f"unknown research features: {', '.join(unknown)}")
    return tuple(BASE_FEATURE_NAMES.index(name) for name in names)


class DirectRidgeModel:
    name = "ridge_direct"
    display_name = "Direct Ridge"
    representation = "current_five_sensor_snapshot"

    def __init__(
        self,
        config: ResearchConfig,
        feature_names: Sequence[str] = BASE_FEATURE_NAMES,
    ):
        self.config = config
        self.feature_names = tuple(feature_names)
        self._indexes = _feature_indexes(self.feature_names)
        self._state: dict[str, Any] | None = None
        self._selected_alpha = float(config.ridge_alpha)
        self._selection_validation_mae: float | None = None

    @property
    def vector_feature_names(self) -> tuple[str, ...]:
        return self.feature_names

    def _vector(self, example: SequenceExample) -> tuple[float, ...]:
        current = example.history[-1]
        return tuple(current[index] for index in self._indexes)

    def fit(self, examples: Sequence[SequenceExample]) -> None:
        self._fit_with_alpha(examples, self._selected_alpha)

    @staticmethod
    def _targets(examples: Sequence[SequenceExample]) -> list[tuple[float, ...]]:
        """Return future temperature changes relative to each forecast origin."""
        return [
            tuple(float(value) - example.current_temperature for value in example.targets)
            for example in examples
        ]

    def _fit_with_alpha(
        self,
        examples: Sequence[SequenceExample],
        alpha: float,
    ) -> None:
        self._state = _fit_multioutput_ridge(
            [self._vector(example) for example in examples],
            self._targets(examples),
            alpha,
        )

    def fit_with_validation(
        self,
        examples: Sequence[SequenceExample],
        validation: Sequence[SequenceExample],
    ) -> None:
        if not examples or not validation:
            self.fit(examples)
            return
        best: tuple[float, float] | None = None
        for alpha in self.config.ridge_alpha_candidates:
            self._fit_with_alpha(examples, alpha)
            absolute_errors = [
                abs(predicted - actual)
                for example in validation
                for predicted, actual in zip(self.predict(example), example.targets)
            ]
            candidate = (fmean(absolute_errors), float(alpha))
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        self._selection_validation_mae, self._selected_alpha = best
        self._fit_with_alpha(examples, self._selected_alpha)

    def refit_on_development(self, examples: Sequence[SequenceExample]) -> None:
        self._fit_with_alpha(examples, self._selected_alpha)

    def predict(self, example: SequenceExample) -> tuple[float, ...]:
        if self._state is None:
            raise ResearchError("ridge model has not been fitted")
        residuals = _predict_multioutput_ridge(self._state, self._vector(example))
        return tuple(example.current_temperature + value for value in residuals)

    def state_dict(self) -> dict[str, Any]:
        if self._state is None:
            raise ResearchError("ridge model has not been fitted")
        return {
            "algorithm": self.name,
            "feature_names": list(self.vector_feature_names),
            "horizons_minutes": list(self.config.horizons_minutes),
            "ridge_alpha": self._selected_alpha,
            "target_transform": "delta_from_current_temperature",
            "selection_validation_mae": self._selection_validation_mae,
            **self._state,
        }

    def training_metadata(self) -> dict[str, Any]:
        return {
            "selection_method": "validation_mae",
            "selected_ridge_alpha": self._selected_alpha,
            "candidate_ridge_alphas": list(self.config.ridge_alpha_candidates),
            "selection_validation_mae": self._selection_validation_mae,
            "target_transform": "delta_from_current_temperature",
        }

    def parameter_count(self) -> int:
        if self._state is None:
            return 0
        return sum(len(output) for output in self._state["coefficients"])


def _linear_slope(values: Sequence[float], spacing_minutes: int) -> float:
    if len(values) < 2:
        return 0.0
    times = [float(index * spacing_minutes) for index in range(len(values))]
    time_mean = fmean(times)
    value_mean = fmean(float(value) for value in values)
    denominator = sum((value - time_mean) ** 2 for value in times)
    if denominator <= 1e-12:
        return 0.0
    return sum(
        (time_value - time_mean) * (float(value) - value_mean)
        for time_value, value in zip(times, values)
    ) / denominator


class HistoryTrendRidgeModel(DirectRidgeModel):
    name = "ridge_history_trend"
    display_name = "Ridge + History/Trend"
    representation = "flattened_history_plus_statistics_and_slope"
    _SUMMARY_NAMES = ("mean", "stddev", "min", "max", "delta", "slope")

    @property
    def vector_feature_names(self) -> tuple[str, ...]:
        labels: list[str] = []
        for name in self.feature_names:
            for index in range(self.config.history_steps):
                minutes_ago = (
                    self.config.history_steps - 1 - index
                ) * self.config.sampling_minutes
                labels.append(f"{name}_t-{minutes_ago}")
            labels.extend(f"{name}_{summary}" for summary in self._SUMMARY_NAMES)
        return tuple(labels)

    def _vector(self, example: SequenceExample) -> tuple[float, ...]:
        values: list[float] = []
        for feature_index in self._indexes:
            history = [row[feature_index] for row in example.history]
            values.extend(history)
            values.extend(
                (
                    fmean(history),
                    pstdev(history),
                    min(history),
                    max(history),
                    history[-1] - history[0],
                    _linear_slope(history, self.config.sampling_minutes),
                )
            )
        return tuple(values)


def horizon_regression_metrics(
    predictions: Sequence[Sequence[float]],
    actuals: Sequence[Sequence[float]],
    horizons_minutes: Sequence[int],
) -> dict[str, Any]:
    """Return pooled and per-horizon regression metrics."""

    horizons = tuple(int(value) for value in horizons_minutes)
    if len(predictions) != len(actuals):
        raise ResearchError("prediction and actual trajectory counts must match")
    for collection in (predictions, actuals):
        if any(len(row) != len(horizons) for row in collection):
            raise ResearchError("trajectory width must match horizons_minutes")
    flattened_predictions = [float(value) for row in predictions for value in row]
    flattened_actuals = [float(value) for row in actuals for value in row]
    actual_summary = descriptive_statistics(flattened_actuals)
    by_horizon: dict[str, Any] = {}
    for index, horizon in enumerate(horizons):
        metrics = regression_metrics(
            [float(row[index]) for row in predictions],
            [float(row[index]) for row in actuals],
        )
        by_horizon[str(horizon)] = {
            "horizon_minutes": horizon,
            **metrics,
        }
    return {
        "overall": regression_metrics(flattened_predictions, flattened_actuals),
        "target_distribution": {
            "count": actual_summary["count"],
            "mean_c": actual_summary["mean"],
            "stddev_c": actual_summary["stddev"],
            "min_c": actual_summary["min"],
            "max_c": actual_summary["max"],
            "range_c": actual_summary["max"] - actual_summary["min"],
            "r2_low_variance_warning": actual_summary["stddev"] < 0.5,
        },
        "by_horizon": by_horizon,
    }


@dataclass(frozen=True, slots=True)
class RiskForecast:
    max_temperature_c: float
    threshold_c: float
    threshold_crossing: bool
    time_to_threshold_minutes: float | None
    heating_rate_c_per_minute: float
    max_adjacent_heating_rate_c_per_minute: float
    risk_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_temperature_c": self.max_temperature_c,
            "threshold_c": self.threshold_c,
            "threshold_crossing": self.threshold_crossing,
            "time_to_threshold_minutes": self.time_to_threshold_minutes,
            "heating_rate_c_per_minute": self.heating_rate_c_per_minute,
            "max_adjacent_heating_rate_c_per_minute": (
                self.max_adjacent_heating_rate_c_per_minute
            ),
            "risk_level": self.risk_level,
        }


def derive_risk(
    temperatures: Sequence[float],
    horizons_minutes: Sequence[int],
    *,
    threshold_c: float,
    current_temperature: float | None = None,
    medium_margin_c: float = 2.0,
    rapid_heating_c_per_minute: float = 0.10,
) -> RiskForecast:
    """Convert a predicted trajectory into maintenance-oriented risk fields."""

    values = tuple(float(value) for value in temperatures)
    horizons = tuple(int(value) for value in horizons_minutes)
    if not values or len(values) != len(horizons):
        raise ResearchError("risk temperatures must match non-empty horizons")
    if any(not math.isfinite(value) for value in values):
        raise ResearchError("risk temperatures must be finite")
    if tuple(sorted(horizons)) != horizons or any(value <= 0 for value in horizons):
        raise ResearchError("risk horizons must be positive and increasing")

    timeline = list(horizons)
    trajectory = list(values)
    if current_temperature is not None:
        current = float(current_temperature)
        if not math.isfinite(current):
            raise ResearchError("current_temperature must be finite")
        timeline.insert(0, 0)
        trajectory.insert(0, current)
    heating_rate = _slope_at_times(timeline, trajectory)
    adjacent_heating_rates = [
        (next_value - previous_value) / (next_time - previous_time)
        for previous_time, previous_value, next_time, next_value in zip(
            timeline,
            trajectory,
            timeline[1:],
            trajectory[1:],
        )
    ]
    max_adjacent_heating_rate = max(adjacent_heating_rates, default=0.0)
    crossing_time: float | None = None
    if trajectory[0] >= threshold_c:
        crossing_time = float(timeline[0])
    else:
        for previous_time, previous_value, next_time, next_value in zip(
            timeline,
            trajectory,
            timeline[1:],
            trajectory[1:],
        ):
            if next_value < threshold_c:
                continue
            if next_value > previous_value:
                fraction = (threshold_c - previous_value) / (
                    next_value - previous_value
                )
                crossing_time = previous_time + fraction * (
                    next_time - previous_time
                )
            else:
                crossing_time = float(next_time)
            break
    max_temperature = max(values)
    crossing = crossing_time is not None
    if crossing:
        risk_level = "high"
    elif (
        max_temperature >= threshold_c - medium_margin_c
        or max(heating_rate, max_adjacent_heating_rate)
        >= rapid_heating_c_per_minute
    ):
        risk_level = "medium"
    else:
        risk_level = "low"
    return RiskForecast(
        max_temperature_c=max_temperature,
        threshold_c=float(threshold_c),
        threshold_crossing=crossing,
        time_to_threshold_minutes=crossing_time,
        heating_rate_c_per_minute=heating_rate,
        max_adjacent_heating_rate_c_per_minute=max_adjacent_heating_rate,
        risk_level=risk_level,
    )


def _slope_at_times(times: Sequence[int], values: Sequence[float]) -> float:
    if len(times) < 2:
        return 0.0
    time_mean = fmean(float(value) for value in times)
    value_mean = fmean(float(value) for value in values)
    denominator = sum((float(value) - time_mean) ** 2 for value in times)
    if denominator <= 1e-12:
        return 0.0
    return sum(
        (float(time_value) - time_mean) * (float(value) - value_mean)
        for time_value, value in zip(times, values)
    ) / denominator


def _event_risk_evaluation(
    predicted_risks: Sequence[RiskForecast],
    examples: Sequence[SequenceExample],
    config: ResearchConfig,
) -> dict[str, Any]:
    """Build one deduplicated observed grid and evaluate warning episodes."""

    observed: dict[tuple[str, datetime], float] = {}
    for example in examples:
        points = ((example.anchor_time, example.current_temperature), *zip(
            example.target_times,
            example.targets,
        ))
        for timestamp, temperature in points:
            key = (example.device_id, timestamp)
            value = float(temperature)
            previous = observed.get(key)
            if previous is not None and not math.isclose(
                previous, value, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ResearchError(
                    "overlapping evaluation windows disagree on observed truth"
                )
            observed[key] = value
    observations = [
        TemperatureObservation(device_id, timestamp, temperature)
        for (device_id, timestamp), temperature in sorted(observed.items())
    ]
    warnings = [
        WarningSignal(
            device_id=example.device_id,
            issued_at=example.anchor_time,
            predicts_threshold_crossing=risk.threshold_crossing,
        )
        for example, risk in zip(examples, predicted_risks)
    ]
    return evaluate_event_warnings(
        observations,
        warnings,
        threshold_c=config.threshold_c,
        sampling_minutes=config.sampling_minutes,
        event_merge_gap_minutes=config.event_merge_gap_minutes,
        warning_lookback_minutes=config.warning_lookback_minutes,
        warning_cooldown_minutes=config.warning_cooldown_minutes,
    )


def _risk_metrics(
    predictions: Sequence[Sequence[float]],
    actuals: Sequence[Sequence[float]],
    examples: Sequence[SequenceExample],
    config: ResearchConfig,
) -> dict[str, Any]:
    predicted_risks = [
        derive_risk(
            values,
            config.horizons_minutes,
            threshold_c=config.threshold_c,
            current_temperature=example.current_temperature,
            medium_margin_c=config.medium_risk_margin_c,
            rapid_heating_c_per_minute=config.rapid_heating_c_per_minute,
        )
        for values, example in zip(predictions, examples)
    ]
    actual_risks = [
        derive_risk(
            values,
            config.horizons_minutes,
            threshold_c=config.threshold_c,
            current_temperature=example.current_temperature,
            medium_margin_c=config.medium_risk_margin_c,
            rapid_heating_c_per_minute=config.rapid_heating_c_per_minute,
        )
        for values, example in zip(actuals, examples)
    ]
    early_warning_indexes = [
        index
        for index, example in enumerate(examples)
        if example.current_temperature < config.threshold_c
    ]
    ongoing_origin_indexes = [
        index
        for index, example in enumerate(examples)
        if example.current_temperature >= config.threshold_c
    ]
    early_warning_index_set = set(early_warning_indexes)
    paired_threshold_times = [
        (predicted.time_to_threshold_minutes, actual.time_to_threshold_minutes)
        for index, (predicted, actual) in enumerate(
            zip(predicted_risks, actual_risks)
        )
        if index in early_warning_index_set
        if predicted.time_to_threshold_minutes is not None
        and actual.time_to_threshold_minutes is not None
    ]
    level_matches = [
        predicted_risks[index].risk_level == actual_risks[index].risk_level
        for index in early_warning_indexes
    ]
    predicted_crossings = [
        predicted_risks[index].threshold_crossing
        for index in early_warning_indexes
    ]
    actual_crossings = [
        actual_risks[index].threshold_crossing
        for index in early_warning_indexes
    ]
    threshold_scores = [
        predicted_risks[index].max_temperature_c - config.threshold_c
        for index in early_warning_indexes
    ]
    crossing_classification = classification_metrics(
        predicted_crossings,
        actual_crossings,
    )
    # A window-level confusion matrix cannot measure event warning lead time.
    # The generic helper's convenience field would otherwise look measured.
    crossing_classification.pop("lead_time_minutes", None)
    threshold_crossing = {
        **crossing_classification,
        **binary_score_metrics(threshold_scores, actual_crossings),
        "sample_count": len(early_warning_indexes),
        "excluded_ongoing_origin_count": len(ongoing_origin_indexes),
        "cohort": "origin_temperature_below_threshold",
        "score_definition": "predicted_max_temperature_c_minus_threshold_c",
        "lead_time_status": "not_computed_without_event_matching",
    }
    ongoing_predicted = [
        max(float(value) for value in predictions[index]) >= config.threshold_c
        for index in ongoing_origin_indexes
    ]
    ongoing_actual = [
        max(float(value) for value in actuals[index]) >= config.threshold_c
        for index in ongoing_origin_indexes
    ]
    ongoing_scores = [
        max(float(value) for value in predictions[index]) - config.threshold_c
        for index in ongoing_origin_indexes
    ]
    ongoing_classification = classification_metrics(
        ongoing_predicted,
        ongoing_actual,
    )
    ongoing_classification.pop("lead_time_minutes", None)
    ongoing_detection = {
        **ongoing_classification,
        **binary_score_metrics(ongoing_scores, ongoing_actual),
        "sample_count": len(ongoing_origin_indexes),
        "cohort": "origin_temperature_at_or_above_threshold",
        "definition": "future_trajectory_at_or_above_threshold",
        "lead_time_status": "not_applicable_to_ongoing_origin",
    }
    return {
        "threshold_crossing": threshold_crossing,
        "ongoing_origin_detection": ongoing_detection,
        "max_temperature": regression_metrics(
            [value.max_temperature_c for value in predicted_risks],
            [value.max_temperature_c for value in actual_risks],
        ),
        "heating_rate": regression_metrics(
            [value.heating_rate_c_per_minute for value in predicted_risks],
            [value.heating_rate_c_per_minute for value in actual_risks],
        ),
        "max_adjacent_heating_rate": regression_metrics(
            [
                value.max_adjacent_heating_rate_c_per_minute
                for value in predicted_risks
            ],
            [
                value.max_adjacent_heating_rate_c_per_minute
                for value in actual_risks
            ],
        ),
        "time_to_threshold": regression_metrics(
            [float(pair[0]) for pair in paired_threshold_times],
            [float(pair[1]) for pair in paired_threshold_times],
        ),
        "risk_level_accuracy": (
            sum(level_matches) / len(level_matches) if level_matches else None
        ),
        "risk_level_accuracy_cohort": "origin_temperature_below_threshold",
        "cohorts": {
            "total_window_count": len(examples),
            "early_warning_window_count": len(early_warning_indexes),
            "ongoing_origin_window_count": len(ongoing_origin_indexes),
        },
        "event_evaluation": _event_risk_evaluation(
            predicted_risks,
            examples,
            config,
        ),
    }


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    summary = descriptive_statistics(values)
    return {
        "unit": "ms_per_sequence",
        "count": summary["count"],
        "mean_ms": summary["mean"],
        "median_ms": summary["median"],
        "p95_ms": summary["p95"],
        "p99_ms": summary["p99"],
        "max_ms": summary["max"],
        "total_ms": sum(values),
    }


def _evaluation_from_predictions(
    predictions: Sequence[Sequence[float]],
    examples: Sequence[SequenceExample],
    config: ResearchConfig,
    latencies_ms: Sequence[float],
    *,
    include_prediction_records: bool = True,
) -> dict[str, Any]:
    actuals = [example.targets for example in examples]
    metrics = horizon_regression_metrics(
        predictions,
        actuals,
        config.horizons_minutes,
    )
    latest: dict[str, Any] | None = None
    if examples:
        example = examples[-1]
        predicted = tuple(float(value) for value in predictions[-1])
        risk = derive_risk(
            predicted,
            config.horizons_minutes,
            threshold_c=config.threshold_c,
            current_temperature=example.current_temperature,
            medium_margin_c=config.medium_risk_margin_c,
            rapid_heating_c_per_minute=config.rapid_heating_c_per_minute,
        )
        latest = {
            "device_id": example.device_id,
            "anchor_time": _iso(example.anchor_time),
            "current_temperature_c": example.current_temperature,
            "trajectory": [
                {
                    "horizon_minutes": horizon,
                    "predicted_temperature_c": predicted[index],
                    "actual_temperature_c": example.targets[index],
                }
                for index, horizon in enumerate(config.horizons_minutes)
            ],
            "risk": risk.to_dict(),
        }
    result = {
        **metrics,
        "risk": _risk_metrics(predictions, actuals, examples, config),
        "latest_forecast": latest,
        "inference_latency_ms": _latency_summary(latencies_ms),
    }
    if include_prediction_records:
        result["prediction_records"] = [
            {
                "sample_id": _sample_id(example, config),
                "device_id": example.device_id,
                "origin_time": _iso(example.anchor_time),
                "target_time": _iso(example.target_times[horizon_index]),
                "horizon_minutes": int(config.horizons_minutes[horizon_index]),
                "actual": float(example.targets[horizon_index]),
                "predicted": float(prediction[horizon_index]),
                "error": (
                    float(prediction[horizon_index])
                    - float(example.targets[horizon_index])
                ),
            }
            for example, prediction in zip(examples, predictions)
            for horizon_index in range(len(config.horizons_minutes))
        ]
    return result


def evaluate_model(
    model: Any,
    examples: Sequence[SequenceExample],
    config: ResearchConfig,
) -> dict[str, Any]:
    """Evaluate an already-fitted adapter and measure each prediction call."""

    if not examples:
        raise ResearchError("model evaluation requires at least one sequence")
    predictions: list[tuple[float, ...]] = []
    latencies: list[float] = []
    for example in examples:
        started = time.perf_counter_ns()
        values = tuple(float(value) for value in model.predict(example))
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if len(values) != len(config.horizons_minutes):
            raise ResearchError("adapter prediction width does not match horizons")
        if not all(math.isfinite(value) for value in values):
            raise ResearchError("adapter returned non-finite predictions")
        predictions.append(values)
        latencies.append(elapsed_ms)
    return _evaluation_from_predictions(predictions, examples, config, latencies)


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    name: str
    factory: Callable[[ResearchConfig, tuple[str, ...]], Any] | None
    display_name: str
    description: str
    required_modules: tuple[str, ...] = ()
    suggested_dependencies: tuple[str, ...] = ()
    unavailable_reason: str | None = None


class ModelRegistry:
    """Registration boundary for built-in and optional model adapters.

    A factory receives ``(ResearchConfig, feature_names)``.  Its adapter must
    implement ``fit(examples)`` and ``predict(example)``.  For efficiency
    reporting it should also implement ``state_dict()`` and may implement
    ``parameter_count()``.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, ModelRegistration] = {}

    def register(
        self,
        name: str,
        factory: Callable[[ResearchConfig, tuple[str, ...]], Any] | None,
        *,
        display_name: str,
        description: str,
        required_modules: Sequence[str] = (),
        suggested_dependencies: Sequence[str] = (),
        unavailable_reason: str | None = None,
        replace: bool = False,
    ) -> None:
        normalized = str(name).strip().lower()
        if not normalized:
            raise ResearchError("model registry names cannot be empty")
        if factory is not None and not callable(factory):
            raise ResearchError("model registry factory must be callable or None")
        if normalized in self._registrations and not replace:
            raise ResearchError(f"model {normalized} is already registered")
        self._registrations[normalized] = ModelRegistration(
            name=normalized,
            factory=factory,
            display_name=display_name,
            description=description,
            required_modules=tuple(required_modules),
            suggested_dependencies=tuple(suggested_dependencies),
            unavailable_reason=unavailable_reason,
        )

    def get(self, name: str) -> ModelRegistration | None:
        return self._registrations.get(str(name).strip().lower())

    @staticmethod
    def _module_exists(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def availability(self, name: str) -> dict[str, Any]:
        registration = self.get(name)
        if registration is None:
            return {
                "name": str(name),
                "status": "unavailable",
                "reason": "model is not registered",
                "required_modules": [],
                "missing_modules": [],
                "suggested_dependencies": [],
            }
        missing = [
            module
            for module in registration.required_modules
            if not self._module_exists(module)
        ]
        if registration.factory is None:
            status = "unavailable"
            reason = registration.unavailable_reason or "model adapter is not registered"
        elif missing:
            status = "unavailable"
            reason = f"missing optional modules: {', '.join(missing)}"
        else:
            status = "available"
            reason = None
        return {
            "name": registration.name,
            "display_name": registration.display_name,
            "description": registration.description,
            "status": status,
            "reason": reason,
            "required_modules": list(registration.required_modules),
            "missing_modules": missing,
            "suggested_dependencies": list(registration.suggested_dependencies),
        }

    def describe(self) -> list[dict[str, Any]]:
        return [self.availability(name) for name in self._registrations]


def build_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        "ridge_direct",
        lambda config, features: DirectRidgeModel(config, features),
        display_name="Direct Ridge",
        description="One direct Ridge output per horizon from the current sensors.",
    )
    registry.register(
        "ridge_history_trend",
        lambda config, features: HistoryTrendRidgeModel(config, features),
        display_name="Ridge + History/Trend",
        description="Direct multi-horizon Ridge over history, summaries, and slopes.",
    )
    optional = {
        "dlinear": ("DLinear", ("torch",)),
        "lstm": ("LSTM", ("torch",)),
        "tcn": ("TCN", ("torch",)),
        "patchtst": ("PatchTST", ("torch",)),
    }
    for name, (display_name, dependencies) in optional.items():
        registry.register(
            name,
            None,
            display_name=display_name,
            description=(
                f"Optional {display_name} adapter for the same sequence contract."
            ),
            suggested_dependencies=dependencies,
            unavailable_reason=(
                "no executable adapter is bundled; install a supported backend "
                "and register an implementation through ModelRegistry.register"
            ),
        )
    return registry


def _model_size_bytes(model: Any) -> int:
    state_method = getattr(model, "state_dict", None)
    if not callable(state_method):
        raise ResearchError("adapter must implement state_dict for size measurement")
    serialized = json.dumps(
        state_method(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return len(serialized)


def _parameter_count(model: Any) -> int | None:
    method = getattr(model, "parameter_count", None)
    return int(method()) if callable(method) else None


def _without_latency(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key != "inference_latency_ms"
    }


def _fit_with_development_validation(
    model: Any,
    training: Sequence[SequenceExample],
    validation: Sequence[SequenceExample],
) -> None:
    """Select training controls without exposing any outer test observations."""
    method = getattr(model, "fit_with_validation", None)
    if callable(method) and validation:
        method(training, validation)
    else:
        model.fit(training)


def _refit_on_development(
    model: Any,
    development: Sequence[SequenceExample],
) -> None:
    """Refit frozen hyperparameters on all data available before the test gap."""
    method = getattr(model, "refit_on_development", None)
    if callable(method):
        method(development)
    else:
        model.fit(development)


def _training_metadata(model: Any) -> dict[str, Any]:
    method = getattr(model, "training_metadata", None)
    return dict(method()) if callable(method) else {
        "selection_method": "no_hyperparameters",
        "target_transform": "identity",
    }


def _inner_fold_development_split(
    examples: Sequence[SequenceExample],
    config: ResearchConfig,
) -> tuple[tuple[SequenceExample, ...], tuple[SequenceExample, ...]]:
    """Make a chronological inner validation tail with its own purge gap."""
    values = tuple(examples)
    gap = config.effective_gap_steps
    validation_count = max(1, int(len(values) * config.validation_fraction))
    training_end = len(values) - validation_count - gap
    if training_end < max(2, config.history_steps):
        return values, ()
    training = values[:training_end]
    validation = values[training_end + gap :]
    if not validation or not _has_safe_boundary(training, validation):
        return values, ()
    return training, validation


def _walk_forward_evaluation(
    registration: ModelRegistration,
    folds: Sequence[WalkForwardFold],
    config: ResearchConfig,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    all_predictions: list[tuple[float, ...]] = []
    all_examples: list[SequenceExample] = []
    all_latencies: list[float] = []
    fold_results: list[dict[str, Any]] = []
    training_time_ms = 0.0
    for fold in folds:
        if registration.factory is None:
            raise ResearchError("walk-forward model adapter is unavailable")
        model = registration.factory(config, feature_names)
        inner_training, inner_validation = _inner_fold_development_split(
            fold.train, config
        )
        started = time.perf_counter_ns()
        _fit_with_development_validation(model, inner_training, inner_validation)
        _refit_on_development(model, fold.train)
        training_time_ms += (time.perf_counter_ns() - started) / 1_000_000
        predictions: list[tuple[float, ...]] = []
        latencies: list[float] = []
        for example in fold.test:
            inference_started = time.perf_counter_ns()
            prediction = tuple(float(value) for value in model.predict(example))
            latencies.append(
                (time.perf_counter_ns() - inference_started) / 1_000_000
            )
            if len(prediction) != len(config.horizons_minutes):
                raise ResearchError("adapter prediction width does not match horizons")
            if not all(math.isfinite(value) for value in prediction):
                raise ResearchError("adapter returned non-finite predictions")
            predictions.append(prediction)
        evaluation = _evaluation_from_predictions(
            predictions,
            fold.test,
            config,
            latencies,
            include_prediction_records=False,
        )
        fold_results.append(
            {
                "fold": fold.index,
                "train_sample_count": len(fold.train),
                "inner_training_sample_count": len(inner_training),
                "inner_validation_sample_count": len(inner_validation),
                "test_sample_count": len(fold.test),
                "training": _training_metadata(model),
                **_without_latency(evaluation),
            }
        )
        all_predictions.extend(predictions)
        all_examples.extend(fold.test)
        all_latencies.extend(latencies)
    aggregate = _evaluation_from_predictions(
        all_predictions,
        all_examples,
        config,
        all_latencies,
        include_prediction_records=False,
    )
    return {
        "fold_count": len(folds),
        "gap_steps": config.effective_gap_steps,
        "gap_minutes": config.effective_gap_steps * config.sampling_minutes,
        "training_time_ms_total": training_time_ms,
        "folds": fold_results,
        "aggregate": aggregate,
    }


@dataclass(frozen=True, slots=True)
class AblationSpec:
    id: str
    label: str
    feature_names: tuple[str, ...]
    model_name: str


DEFAULT_ABLATIONS = (
    AblationSpec("A", "Temperature", ("temperature",), "ridge_direct"),
    AblationSpec(
        "B",
        "Temperature + Humidity",
        ("temperature", "humidity"),
        "ridge_direct",
    ),
    AblationSpec(
        "C",
        "Temperature + Acceleration",
        ("temperature", "accel_x", "accel_y", "accel_z"),
        "ridge_direct",
    ),
    AblationSpec("D", "All Five Sensors", BASE_FEATURE_NAMES, "ridge_direct"),
    AblationSpec(
        "E",
        "All Five + Historical Trend",
        BASE_FEATURE_NAMES,
        "ridge_history_trend",
    ),
)


def run_feature_ablations(
    split: DatasetSplit,
    config: ResearchConfig,
    *,
    specs: Sequence[AblationSpec] = DEFAULT_ABLATIONS,
) -> list[dict[str, Any]]:
    """Run the five predeclared multi-sensor feature comparisons."""

    registry = build_default_registry()
    results: list[dict[str, Any]] = []
    for spec in specs:
        registration = registry.get(spec.model_name)
        base = {
            "id": spec.id,
            "label": spec.label,
            "features": list(spec.feature_names),
            "model": spec.model_name,
            "model_display_name": (
                registration.display_name
                if registration is not None
                else spec.model_name
            ),
            "evaluation_scope": "locked_test",
        }
        if registration is None or registration.factory is None:
            results.append({**base, "status": "unavailable"})
            continue
        try:
            model = registration.factory(config, spec.feature_names)
            started = time.perf_counter_ns()
            _fit_with_development_validation(model, split.train, split.validation)
            validation = evaluate_model(model, split.validation, config)
            _refit_on_development(model, (*split.train, *split.validation))
            training_ms = (time.perf_counter_ns() - started) / 1_000_000
            test = evaluate_model(model, split.test, config)
            validation.pop("prediction_records", None)
            test.pop("prediction_records", None)
            results.append(
                {
                    **base,
                    "status": "available",
                    "validation": validation,
                    "test": test,
                    "training": _training_metadata(model),
                    "efficiency": {
                        "training_time_ms": training_ms,
                        "inference_latency_ms": test["inference_latency_ms"],
                        "model_size_bytes": _model_size_bytes(model),
                        "parameter_count": _parameter_count(model),
                    },
                }
            )
        except (ResearchError, TypeError, ValueError) as error:
            results.append(
                {
                    **base,
                    "status": "failed",
                    "reason": str(error),
                    "error_type": type(error).__name__,
                }
            )
    return results


def _block_mae(prediction_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate overlapping horizon errors at the device-day block level."""

    grouped: dict[str, list[float]] = {}
    for record in prediction_records:
        origin = str(record["origin_time"])
        block_id = f"{record['device_id']}:{origin[:10]}"
        grouped.setdefault(block_id, []).append(abs(float(record["error"])))
    return {
        block_id: fmean(values)
        for block_id, values in sorted(grouped.items())
    }


def _paired_model_statistics(
    models: Mapping[str, Mapping[str, Any]],
    config: ResearchConfig,
) -> list[dict[str, Any]]:
    """Compare each locked-test candidate with Direct Ridge by paired day."""

    reference = models.get("ridge_direct")
    reference_records = (
        reference.get("test", {}).get("prediction_records")
        if isinstance(reference, Mapping)
        else None
    )
    if not isinstance(reference_records, list):
        return [
            {
                "comparison": "candidate_vs_ridge_direct",
                "status": "unavailable",
                "reason": "Direct Ridge locked-test prediction ledger is unavailable",
            }
        ]
    reference_blocks = _block_mae(reference_records)
    results: list[dict[str, Any]] = []
    for model_name, model in sorted(models.items()):
        if model_name == "ridge_direct" or not isinstance(model, Mapping):
            continue
        records = model.get("test", {}).get("prediction_records")
        if not isinstance(records, list):
            results.append(
                {
                    "comparison": f"{model_name}_vs_ridge_direct",
                    "reference_model": "ridge_direct",
                    "candidate_model": model_name,
                    "status": "unavailable",
                    "reason": model.get("reason") or "locked-test ledger unavailable",
                }
            )
            continue
        statistics = paired_block_bootstrap(
            reference_blocks,
            _block_mae(records),
            n_resamples=config.bootstrap_repetitions,
            seed=config.bootstrap_seed,
        )
        results.append(
            {
                "comparison": f"{model_name}_vs_ridge_direct",
                "reference_model": "ridge_direct",
                "candidate_model": model_name,
                "block_definition": "device_id_x_origin_utc_date",
                **statistics,
            }
        )
    return results


def _attach_direct_ridge_skill(models: Mapping[str, Any]) -> None:
    """Add an interpretable score without changing or suppressing raw metrics."""
    reference = models.get("ridge_direct")
    if not isinstance(reference, Mapping):
        return
    for scope in ("validation", "test", "cross_device"):
        reference_scope = reference.get(scope)
        if not isinstance(reference_scope, Mapping):
            continue
        reference_mae = reference_scope.get("overall", {}).get("mae")
        if not isinstance(reference_mae, (int, float)) or reference_mae <= 0:
            continue
        for result in models.values():
            if not isinstance(result, dict):
                continue
            evaluation = result.get(scope)
            if not isinstance(evaluation, dict):
                continue
            candidate_mae = evaluation.get("overall", {}).get("mae")
            if isinstance(candidate_mae, (int, float)):
                evaluation["skill_score_vs_ridge_direct"] = (
                    1.0 - float(candidate_mae) / float(reference_mae)
                )


def run_research_experiment(
    training_records: Iterable[Any],
    evaluation_records: Iterable[Any] | None = None,
    *,
    config: ResearchConfig | None = None,
    model_names: Sequence[str] | None = None,
    registry: ModelRegistry | None = None,
    include_ablations: bool = True,
) -> dict[str, Any]:
    """Execute a leakage-audited, JSON-serializable comparison experiment."""

    config = config or ResearchConfig()
    registry = registry or build_default_registry()
    requested_models = tuple(model_names or DEFAULT_MODEL_NAMES)
    training_dataset = build_sequence_dataset(training_records, config)
    split = chronological_split(training_dataset, config)
    # Inner walk-forward validation is restricted to the outer development
    # period.  In particular, this excludes both the validation/test boundary
    # embargo and every locked-test sequence.
    development_examples = tuple(
        example
        for example in training_dataset.examples
        if example.anchor_time <= split.validation[-1].anchor_time
    )
    locked_test_first_anchor = split.test[0].anchor_time
    if any(
        target_time >= locked_test_first_anchor
        for example in development_examples
        for target_time in example.target_times
    ):
        raise ResearchError(
            "development target crosses the locked-test boundary"
        )
    folds = walk_forward_splits(
        training_dataset,
        config,
        examples=development_examples,
        n_splits=config.walk_forward_folds,
    )
    walk_forward_targets_are_locked = all(
        target_time < locked_test_first_anchor
        for fold in folds
        for example in (*fold.train, *fold.test)
        for target_time in example.target_times
    )
    if not walk_forward_targets_are_locked:
        raise ResearchError("walk-forward fold crossed the locked-test boundary")
    evaluation_dataset = (
        build_sequence_dataset(evaluation_records, config)
        if evaluation_records is not None
        else None
    )

    assigned_splits = {
        _sample_id(example, config): partition
        for partition, examples in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for example in examples
    }
    split_manifest_records = [
        {
            "sample_id": _sample_id(example, config),
            "device_id": example.device_id,
            "origin_time": _iso(example.anchor_time),
            "history_start_time": _iso(example.history_times[0]),
            "history_end_time": _iso(example.history_times[-1]),
            "first_target_time": _iso(example.target_times[0]),
            "last_target_time": _iso(example.target_times[-1]),
            "split": assigned_splits.get(_sample_id(example, config), "purged_gap"),
        }
        for example in training_dataset.examples
    ]
    if evaluation_dataset is not None:
        split_manifest_records.extend(
            {
                "sample_id": _sample_id(example, config),
                "device_id": example.device_id,
                "origin_time": _iso(example.anchor_time),
                "history_start_time": _iso(example.history_times[0]),
                "history_end_time": _iso(example.history_times[-1]),
                "first_target_time": _iso(example.target_times[0]),
                "last_target_time": _iso(example.target_times[-1]),
                "split": "external_test",
            }
            for example in evaluation_dataset.examples
        )

    model_results: dict[str, Any] = {}
    for requested_name in requested_models:
        name = str(requested_name).strip().lower()
        availability = registry.availability(name)
        registration = registry.get(name)
        if availability["status"] != "available" or registration is None:
            model_results[name] = availability
            continue
        feature_names = BASE_FEATURE_NAMES
        try:
            if registration.factory is None:
                raise ResearchError("model factory is unavailable")
            model = registration.factory(config, feature_names)
            started = time.perf_counter_ns()
            _fit_with_development_validation(model, split.train, split.validation)
            validation = evaluate_model(model, split.validation, config)
            selection_training = _training_metadata(model)
            _refit_on_development(model, development_examples)
            training_ms = (time.perf_counter_ns() - started) / 1_000_000
            test = evaluate_model(model, split.test, config)
            cross_device = None
            if evaluation_dataset is not None:
                if not evaluation_dataset.examples:
                    cross_device = {
                        "status": "unavailable",
                        "reason": "evaluation device has no complete sequences",
                    }
                else:
                    evaluated = evaluate_model(
                        model, evaluation_dataset.examples, config
                    )
                    cross_device = {
                        "status": "available",
                        "training_device_ids": list(training_dataset.device_ids),
                        "evaluation_device_ids": list(evaluation_dataset.device_ids),
                        "is_cross_device": (
                            set(training_dataset.device_ids)
                            != set(evaluation_dataset.device_ids)
                        ),
                        **evaluated,
                    }
            result = {
                **availability,
                "feature_names": list(getattr(model, "feature_names", feature_names)),
                "representation": getattr(model, "representation", "custom_adapter"),
                "training": {
                    **selection_training,
                    "selection_fit_sample_count": len(split.train),
                    "selection_validation_sample_count": len(split.validation),
                    "final_refit_sample_count": len(development_examples),
                    "locked_test_used_for_selection": False,
                },
                "validation": validation,
                "test": test,
                "walk_forward": _walk_forward_evaluation(
                    registration,
                    folds,
                    config,
                    feature_names,
                ),
                "efficiency": {
                    "training_time_ms": training_ms,
                    "inference_latency_ms": test["inference_latency_ms"],
                    "model_size_bytes": _model_size_bytes(model),
                    "parameter_count": _parameter_count(model),
                },
            }
            if cross_device is not None:
                result["cross_device"] = cross_device
            model_results[name] = result
        except Exception as error:  # Adapter failures belong in results, not fake metrics.
            model_results[name] = {
                **availability,
                "status": "failed",
                "reason": str(error),
                "error_type": type(error).__name__,
            }

    _attach_direct_ridge_skill(model_results)
    statistical_comparisons = _paired_model_statistics(model_results, config)
    return {
        "schema_version": 2,
        "configuration": config.to_dict(),
        "methodology": {
            "task": "direct_multi_horizon_temperature_trajectory_forecasting",
            "input": (
                f"{config.history_steps} readings x {len(BASE_FEATURE_NAMES)} "
                "sensor features"
            ),
            "target": list(config.horizons_minutes),
            "learned_model_target_transform": (
                "future_temperature_minus_current_temperature; prediction is "
                "converted back to absolute °C before scoring"
            ),
            "alignment": (
                "exact relative timestamps; no interpolation or nearest-future "
                "target matching"
            ),
            "normalization": "fit on training partition only for each fitted model",
            "split": "chronological train/validation/test with purged gaps",
            "walk_forward": (
                "development-only expanding training window with the same purged "
                "gap; the locked test and its boundary embargo are excluded"
            ),
            "model_selection_scope": (
                "hyperparameters and early stopping use validation data only; "
                "frozen settings are refit on the complete development period; "
                "the locked test is reserved for one final holdout evaluation"
            ),
            "leakage_controls": [
                "records are never shuffled",
                "each earlier partition's latest label precedes the next anchor",
                "walk-forward folds and labels end before the locked-test anchor",
                "evaluation-device rows are never passed to model.fit",
                "locked-test errors never select alpha, tree count, or epoch count",
                "duplicated timestamps are excluded instead of arbitrarily selected",
            ],
        },
        "dataset": {
            "training": training_dataset.to_dict(),
            "split": split.to_dict(config.sampling_minutes),
            "walk_forward_splits": [
                fold.to_dict(config.sampling_minutes) for fold in folds
            ],
            "walk_forward_scope": {
                "scope": "development_only",
                "development_sequence_count": len(development_examples),
                "last_development_anchor_time": _iso(
                    development_examples[-1].anchor_time
                ),
                "locked_test_first_anchor_time": _iso(locked_test_first_anchor),
                "all_fold_targets_end_before_locked_test_anchor": (
                    walk_forward_targets_are_locked
                ),
            },
            "evaluation": (
                evaluation_dataset.to_dict()
                if evaluation_dataset is not None
                else None
            ),
            "split_manifest_records": split_manifest_records,
        },
        "model_capabilities": registry.describe(),
        "models": model_results,
        "statistical_comparisons": statistical_comparisons,
        "ablations": (
            run_feature_ablations(split, config) if include_ablations else []
        ),
    }


__all__ = [
    "AblationSpec",
    "AlignmentQuality",
    "BASE_FEATURE_NAMES",
    "DEFAULT_ABLATIONS",
    "DEFAULT_HORIZONS_MINUTES",
    "DEFAULT_MODEL_NAMES",
    "DatasetSplit",
    "DirectRidgeModel",
    "HistoryTrendRidgeModel",
    "MAX_HORIZON_MINUTES",
    "InferenceSequence",
    "ModelRegistry",
    "OPTIONAL_MODEL_NAMES",
    "ResearchConfig",
    "ResearchError",
    "RiskForecast",
    "SequenceDataset",
    "SequenceExample",
    "WalkForwardFold",
    "build_default_registry",
    "build_inference_sequence",
    "build_sequence_dataset",
    "chronological_split",
    "derive_risk",
    "evaluate_model",
    "horizon_regression_metrics",
    "run_feature_ablations",
    "run_research_experiment",
    "walk_forward_splits",
]
