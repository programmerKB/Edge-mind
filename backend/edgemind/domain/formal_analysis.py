"""Pure, deterministic event and paired-statistics rules for formal studies.

This module deliberately knows nothing about databases, HTTP, model adapters,
or report files.  Callers supply an observed temperature grid and positive or
negative warning signals; the returned dictionaries contain only JSON-ready
values and can therefore be persisted by any outer-layer workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import random
from statistics import fmean, median
from typing import Any, Mapping, Sequence


class FormalAnalysisError(ValueError):
    """Raised when a formal-analysis contract is ambiguous or invalid."""


@dataclass(frozen=True, slots=True)
class TemperatureObservation:
    """One measured temperature on the evaluation grid."""

    device_id: str
    recorded_at: datetime
    temperature_c: float


@dataclass(frozen=True, slots=True)
class WarningSignal:
    """One model decision emitted at an observed forecast origin."""

    device_id: str
    issued_at: datetime
    predicts_threshold_crossing: bool = True


@dataclass(frozen=True, slots=True)
class _Event:
    device_id: str
    start_time: datetime
    end_time: datetime
    peak_temperature_c: float
    positive_observation_count: int
    censor_reasons: tuple[str, ...]

    @property
    def boundary_censored(self) -> bool:
        return bool(self.censor_reasons)


@dataclass(frozen=True, slots=True)
class _WarningEpisode:
    device_id: str
    warning_times: tuple[datetime, ...]

    @property
    def start_time(self) -> datetime:
        return self.warning_times[0]

    @property
    def end_time(self) -> datetime:
        return self.warning_times[-1]


def _as_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise FormalAnalysisError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise FormalAnalysisError(f"{field} must include an explicit timezone")
    return value.astimezone(timezone.utc)


def _finite(value: float, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FormalAnalysisError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise FormalAnalysisError(f"{field} must be finite")
    return result


def _validated_policy(
    *,
    threshold_c: float,
    sampling_minutes: int,
    event_merge_gap_minutes: int,
    warning_lookback_minutes: int,
    warning_cooldown_minutes: int,
) -> tuple[float, int, int, int, int]:
    threshold = _finite(threshold_c, "threshold_c")
    values = (
        sampling_minutes,
        event_merge_gap_minutes,
        warning_lookback_minutes,
        warning_cooldown_minutes,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise FormalAnalysisError("event policy minute values must be integers")
    if sampling_minutes <= 0:
        raise FormalAnalysisError("sampling_minutes must be positive")
    if event_merge_gap_minutes < 0:
        raise FormalAnalysisError("event_merge_gap_minutes cannot be negative")
    if warning_lookback_minutes <= 0:
        raise FormalAnalysisError("warning_lookback_minutes must be positive")
    if warning_cooldown_minutes < 0:
        raise FormalAnalysisError("warning_cooldown_minutes cannot be negative")
    for name, value in (
        ("event_merge_gap_minutes", event_merge_gap_minutes),
        ("warning_lookback_minutes", warning_lookback_minutes),
        ("warning_cooldown_minutes", warning_cooldown_minutes),
    ):
        if value % sampling_minutes:
            raise FormalAnalysisError(
                f"{name} must be an exact multiple of sampling_minutes"
            )
    return (
        threshold,
        sampling_minutes,
        event_merge_gap_minutes,
        warning_lookback_minutes,
        warning_cooldown_minutes,
    )


def _normalized_observations(
    observations: Sequence[TemperatureObservation],
) -> dict[str, tuple[TemperatureObservation, ...]]:
    by_device: dict[str, list[TemperatureObservation]] = {}
    seen: set[tuple[str, datetime]] = set()
    for observation in observations:
        device_id = str(observation.device_id).strip()
        if not device_id:
            raise FormalAnalysisError("observation device_id cannot be empty")
        recorded_at = _as_utc(observation.recorded_at, "recorded_at")
        key = (device_id, recorded_at)
        if key in seen:
            raise FormalAnalysisError(
                f"duplicate observation grid point: {device_id} {recorded_at.isoformat()}"
            )
        seen.add(key)
        by_device.setdefault(device_id, []).append(
            TemperatureObservation(
                device_id=device_id,
                recorded_at=recorded_at,
                temperature_c=_finite(observation.temperature_c, "temperature_c"),
            )
        )
    if not by_device:
        raise FormalAnalysisError("event evaluation requires observations")
    return {
        device_id: tuple(sorted(rows, key=lambda row: row.recorded_at))
        for device_id, rows in sorted(by_device.items())
    }


def _primitive_events(
    rows: Sequence[TemperatureObservation],
    *,
    threshold_c: float,
    step: timedelta,
) -> list[_Event]:
    """Build contiguous positive runs while recording uncertain boundaries."""
    events: list[_Event] = []
    index = 0
    while index < len(rows):
        if rows[index].temperature_c < threshold_c:
            index += 1
            continue
        start_index = index
        while (
            index + 1 < len(rows)
            and rows[index + 1].recorded_at - rows[index].recorded_at == step
            and rows[index + 1].temperature_c >= threshold_c
        ):
            index += 1
        end_index = index
        reasons: list[str] = []
        if start_index == 0:
            reasons.append("onset_not_observed_before_evaluation_start")
        elif rows[start_index].recorded_at - rows[start_index - 1].recorded_at != step:
            reasons.append("onset_not_observed_after_internal_gap")
        if end_index == len(rows) - 1:
            reasons.append("offset_not_observed_before_evaluation_end")
        elif rows[end_index + 1].recorded_at - rows[end_index].recorded_at != step:
            reasons.append("offset_not_observed_before_internal_gap")
        selected = rows[start_index : end_index + 1]
        events.append(
            _Event(
                device_id=selected[0].device_id,
                start_time=selected[0].recorded_at,
                end_time=selected[-1].recorded_at,
                peak_temperature_c=max(row.temperature_c for row in selected),
                positive_observation_count=len(selected),
                censor_reasons=tuple(reasons),
            )
        )
        index += 1
    return events


def _complete_grid_between(
    observed_times: set[datetime],
    start: datetime,
    end: datetime,
    step: timedelta,
) -> bool:
    current = start
    while current <= end:
        if current not in observed_times:
            return False
        current += step
    return True


def _events_for_device(
    rows: Sequence[TemperatureObservation],
    *,
    threshold_c: float,
    sampling_minutes: int,
    event_merge_gap_minutes: int,
    warning_lookback_minutes: int,
) -> list[_Event]:
    step = timedelta(minutes=sampling_minutes)
    primitives = _primitive_events(rows, threshold_c=threshold_c, step=step)
    observed_times = {row.recorded_at for row in rows}
    merged: list[_Event] = []
    for event in primitives:
        if not merged:
            merged.append(event)
            continue
        previous = merged[-1]
        below_threshold_gap = (
            event.start_time - previous.end_time - step
        ).total_seconds() / 60
        grid_is_complete = _complete_grid_between(
            observed_times,
            previous.end_time,
            event.start_time,
            step,
        )
        if 0 <= below_threshold_gap <= event_merge_gap_minutes and grid_is_complete:
            merged[-1] = _Event(
                device_id=previous.device_id,
                start_time=previous.start_time,
                end_time=event.end_time,
                peak_temperature_c=max(
                    previous.peak_temperature_c, event.peak_temperature_c
                ),
                positive_observation_count=(
                    previous.positive_observation_count
                    + event.positive_observation_count
                ),
                censor_reasons=tuple(
                    dict.fromkeys(previous.censor_reasons + event.censor_reasons)
                ),
            )
        else:
            merged.append(event)

    lookback = timedelta(minutes=warning_lookback_minutes)
    completed: list[_Event] = []
    for event in merged:
        reasons = list(event.censor_reasons)
        lookback_start = event.start_time - lookback
        if lookback_start < rows[0].recorded_at:
            reasons.append("incomplete_warning_lookback_at_evaluation_start")
        elif not _complete_grid_between(
            observed_times,
            lookback_start,
            event.start_time,
            step,
        ):
            reasons.append("incomplete_warning_lookback_after_internal_gap")
        completed.append(
            _Event(
                device_id=event.device_id,
                start_time=event.start_time,
                end_time=event.end_time,
                peak_temperature_c=event.peak_temperature_c,
                positive_observation_count=event.positive_observation_count,
                censor_reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return completed


def _event_contains(events: Sequence[_Event], timestamp: datetime) -> bool:
    return any(event.start_time <= timestamp <= event.end_time for event in events)


def _event_between(
    events: Sequence[_Event], start: datetime, end: datetime
) -> bool:
    return any(event.start_time <= end and event.end_time >= start for event in events)


def _warning_episodes(
    times_by_device: Mapping[str, Sequence[datetime]],
    events_by_device: Mapping[str, Sequence[_Event]],
    *,
    cooldown_minutes: int,
) -> list[_WarningEpisode]:
    cooldown = timedelta(minutes=cooldown_minutes)
    episodes: list[_WarningEpisode] = []
    for device_id in sorted(times_by_device):
        times = sorted(set(times_by_device[device_id]))
        groups: list[list[datetime]] = []
        for issued_at in times:
            if (
                groups
                and issued_at - groups[-1][-1] <= cooldown
                and not _event_between(
                    events_by_device.get(device_id, ()),
                    groups[-1][-1],
                    issued_at,
                )
            ):
                groups[-1].append(issued_at)
            else:
                groups.append([issued_at])
        episodes.extend(
            _WarningEpisode(device_id, tuple(group)) for group in groups
        )
    return sorted(episodes, key=lambda item: (item.device_id, item.start_time))


def _event_record(event: _Event, event_id: str, matched: bool) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "device_id": event.device_id,
        "start_time": event.start_time.isoformat(),
        "end_time": event.end_time.isoformat(),
        "peak_temperature_c": event.peak_temperature_c,
        "positive_observation_count": event.positive_observation_count,
        "boundary_censored": event.boundary_censored,
        "censor_reasons": list(event.censor_reasons),
        "matched": matched if not event.boundary_censored else None,
    }


def _lead_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "values": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "values": ordered,
        "mean": fmean(ordered),
        "median": median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def evaluate_event_warnings(
    observations: Sequence[TemperatureObservation],
    warnings: Sequence[WarningSignal],
    *,
    threshold_c: float,
    sampling_minutes: int = 5,
    event_merge_gap_minutes: int = 10,
    warning_lookback_minutes: int = 30,
    warning_cooldown_minutes: int = 5,
) -> dict[str, Any]:
    """Evaluate threshold warnings against one-to-one overheat events.

    An observed grid slot contributes ``sampling_minutes`` to the exposure
    denominator, so acquisition gaps are not mislabeled as operating time.
    Positive signals at an already-hot origin or inside a merged event are not
    early warnings.  Unmatched episodes are false alarms only when their whole
    follow-up window was observed; otherwise they are boundary-censored.
    """

    (
        threshold,
        sampling_minutes,
        event_merge_gap_minutes,
        warning_lookback_minutes,
        warning_cooldown_minutes,
    ) = _validated_policy(
        threshold_c=threshold_c,
        sampling_minutes=sampling_minutes,
        event_merge_gap_minutes=event_merge_gap_minutes,
        warning_lookback_minutes=warning_lookback_minutes,
        warning_cooldown_minutes=warning_cooldown_minutes,
    )
    rows_by_device = _normalized_observations(observations)
    events_by_device = {
        device_id: tuple(
            _events_for_device(
                rows,
                threshold_c=threshold,
                sampling_minutes=sampling_minutes,
                event_merge_gap_minutes=event_merge_gap_minutes,
                warning_lookback_minutes=warning_lookback_minutes,
            )
        )
        for device_id, rows in rows_by_device.items()
    }
    all_events = [
        event
        for device_id in sorted(events_by_device)
        for event in events_by_device[device_id]
    ]
    event_ids = {
        id(event): f"{event.device_id}:event:{event.start_time.isoformat()}"
        for event in all_events
    }
    observation_lookup = {
        (device_id, row.recorded_at): row
        for device_id, rows in rows_by_device.items()
        for row in rows
    }

    input_signal_count = len(warnings)
    positive_signal_count = 0
    ongoing_origin_signal_count = 0
    during_event_signal_count = 0
    early_times: dict[str, list[datetime]] = {}
    for warning in warnings:
        device_id = str(warning.device_id).strip()
        issued_at = _as_utc(warning.issued_at, "issued_at")
        if not device_id:
            raise FormalAnalysisError("warning device_id cannot be empty")
        observation = observation_lookup.get((device_id, issued_at))
        if observation is None:
            raise FormalAnalysisError(
                "every warning must reference an observed device/time origin"
            )
        if not bool(warning.predicts_threshold_crossing):
            continue
        positive_signal_count += 1
        if observation.temperature_c >= threshold:
            ongoing_origin_signal_count += 1
            continue
        if _event_contains(events_by_device[device_id], issued_at):
            during_event_signal_count += 1
            continue
        early_times.setdefault(device_id, []).append(issued_at)

    episodes = _warning_episodes(
        early_times,
        events_by_device,
        cooldown_minutes=warning_cooldown_minutes,
    )
    eligible_events = [event for event in all_events if not event.boundary_censored]
    unmatched_event_ids = {id(event) for event in eligible_events}
    lookback = timedelta(minutes=warning_lookback_minutes)
    matches: list[dict[str, Any]] = []
    episode_matches: dict[int, tuple[_Event, datetime, float]] = {}
    for episode_index, episode in enumerate(episodes):
        candidates: list[tuple[_Event, datetime]] = []
        for event in eligible_events:
            if id(event) not in unmatched_event_ids or event.device_id != episode.device_id:
                continue
            eligible_times = [
                issued_at
                for issued_at in episode.warning_times
                if event.start_time - lookback <= issued_at < event.start_time
            ]
            if eligible_times:
                candidates.append((event, min(eligible_times)))
        if not candidates:
            continue
        event, issued_at = min(candidates, key=lambda item: item[0].start_time)
        lead_minutes = (event.start_time - issued_at).total_seconds() / 60
        unmatched_event_ids.remove(id(event))
        episode_matches[episode_index] = (event, issued_at, lead_minutes)
        matches.append(
            {
                "event_id": event_ids[id(event)],
                "warning_episode_id": (
                    f"{episode.device_id}:warning:{episode.start_time.isoformat()}"
                ),
                "warning_time": issued_at.isoformat(),
                "event_start_time": event.start_time.isoformat(),
                "lead_time_minutes": lead_minutes,
            }
        )

    censored_episode_reasons: dict[int, list[str]] = {}
    false_episode_indexes: list[int] = []
    step = timedelta(minutes=sampling_minutes)
    for index, episode in enumerate(episodes):
        if index in episode_matches:
            continue
        reasons: list[str] = []
        censored_events = [
            event
            for event in all_events
            if event.boundary_censored and event.device_id == episode.device_id
        ]
        if any(
            event.start_time - lookback <= issued_at < event.start_time
            for event in censored_events
            for issued_at in episode.warning_times
        ):
            reasons.append("could_match_boundary_censored_event")
        rows = rows_by_device[episode.device_id]
        observed_times = {row.recorded_at for row in rows}
        follow_up_end = episode.start_time + lookback
        if follow_up_end > rows[-1].recorded_at:
            reasons.append("incomplete_follow_up_at_evaluation_end")
        elif not _complete_grid_between(
            observed_times,
            episode.start_time,
            follow_up_end,
            step,
        ):
            reasons.append("incomplete_follow_up_after_internal_gap")
        if reasons:
            censored_episode_reasons[index] = list(dict.fromkeys(reasons))
        else:
            false_episode_indexes.append(index)

    lead_times = [match[2] for match in episode_matches.values()]
    matched_event_ids = {id(match[0]) for match in episode_matches.values()}
    evaluated_episode_count = len(episode_matches) + len(false_episode_indexes)
    observed_grid_count = sum(len(rows) for rows in rows_by_device.values())
    observed_minutes = observed_grid_count * sampling_minutes
    observed_days = observed_minutes / (24 * 60)
    precision = (
        len(episode_matches) / evaluated_episode_count
        if evaluated_episode_count
        else None
    )
    recall = (
        len(matched_event_ids) / len(eligible_events)
        if eligible_events
        else None
    )
    episode_records = []
    for index, episode in enumerate(episodes):
        if index in episode_matches:
            disposition = "matched"
            reasons: list[str] = []
        elif index in censored_episode_reasons:
            disposition = "boundary_censored"
            reasons = censored_episode_reasons[index]
        else:
            disposition = "false_alarm"
            reasons = []
        episode_records.append(
            {
                "warning_episode_id": (
                    f"{episode.device_id}:warning:{episode.start_time.isoformat()}"
                ),
                "device_id": episode.device_id,
                "start_time": episode.start_time.isoformat(),
                "end_time": episode.end_time.isoformat(),
                "signal_count": len(episode.warning_times),
                "signal_times": [value.isoformat() for value in episode.warning_times],
                "disposition": disposition,
                "censor_reasons": reasons,
            }
        )

    return {
        "status": "available",
        "policy": {
            "threshold_c": threshold,
            "sampling_minutes": sampling_minutes,
            "event_merge_gap_minutes": event_merge_gap_minutes,
            "warning_lookback_minutes": warning_lookback_minutes,
            "warning_cooldown_minutes": warning_cooldown_minutes,
            "event_matching": "chronological_one_event_per_warning_episode",
            "observed_time_definition": "unique_grid_slots_x_sampling_minutes",
        },
        "coverage": {
            "device_count": len(rows_by_device),
            "observed_grid_count": observed_grid_count,
            "observed_minutes": observed_minutes,
            "observed_days": observed_days,
        },
        "events": {
            "detected_count": len(all_events),
            "eligible_count": len(eligible_events),
            "boundary_censored_count": len(all_events) - len(eligible_events),
            "matched_count": len(matched_event_ids),
            "missed_count": len(eligible_events) - len(matched_event_ids),
            "records": [
                _event_record(event, event_ids[id(event)], id(event) in matched_event_ids)
                for event in all_events
            ],
        },
        "warnings": {
            "input_signal_count": input_signal_count,
            "positive_signal_count": positive_signal_count,
            "ongoing_origin_signal_count": ongoing_origin_signal_count,
            "during_event_signal_count": during_event_signal_count,
            "early_warning_episode_count": len(episodes),
            "evaluated_episode_count": evaluated_episode_count,
            "boundary_censored_episode_count": len(censored_episode_reasons),
            "matched_episode_count": len(episode_matches),
            "false_alarm_count": len(false_episode_indexes),
            "records": episode_records,
        },
        "metrics": {
            "event_precision": precision,
            "event_recall": recall,
            "false_alarms_per_observed_day": (
                len(false_episode_indexes) / observed_days
                if observed_days > 0
                else None
            ),
            "lead_time_minutes": _lead_summary(lead_times),
        },
        "matches": matches,
    }


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_block_bootstrap(
    reference_by_block: Mapping[str, float],
    candidate_by_block: Mapping[str, float],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 17,
    minimum_blocks: int = 2,
) -> dict[str, Any]:
    """Bootstrap paired block-level errors with one shared resampling draw.

    Negative ``candidate_minus_reference`` values mean that the candidate has
    lower error.  ``relative_improvement_percent`` uses the opposite sign so a
    positive value means improvement.
    """

    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise FormalAnalysisError("n_resamples must be a positive integer")
    if not 0 < confidence_level < 1:
        raise FormalAnalysisError("confidence_level must be between zero and one")
    if isinstance(minimum_blocks, bool) or not isinstance(minimum_blocks, int):
        raise FormalAnalysisError("minimum_blocks must be an integer")
    if minimum_blocks < 2:
        raise FormalAnalysisError("minimum_blocks must be at least two")
    reference_keys = {str(key) for key in reference_by_block}
    candidate_keys = {str(key) for key in candidate_by_block}
    if reference_keys != candidate_keys:
        missing_candidate = sorted(reference_keys - candidate_keys)
        missing_reference = sorted(candidate_keys - reference_keys)
        raise FormalAnalysisError(
            "paired bootstrap requires identical block IDs; "
            f"missing_candidate={missing_candidate}, "
            f"missing_reference={missing_reference}"
        )
    # Normalizing keys to strings also prevents insertion-order differences
    # from changing a seeded result.
    reference = {
        str(key): _finite(value, "reference block value")
        for key, value in reference_by_block.items()
    }
    candidate = {
        str(key): _finite(value, "candidate block value")
        for key, value in candidate_by_block.items()
    }
    keys = sorted(reference)
    reference_mean = fmean(reference.values()) if reference else None
    candidate_mean = fmean(candidate.values()) if candidate else None
    point_difference = (
        candidate_mean - reference_mean
        if reference_mean is not None and candidate_mean is not None
        else None
    )
    point_relative = (
        (reference_mean - candidate_mean) / abs(reference_mean) * 100
        if reference_mean is not None
        and candidate_mean is not None
        and not math.isclose(reference_mean, 0.0, abs_tol=1e-12)
        else None
    )
    base = {
        "block_count": len(keys),
        "minimum_blocks": minimum_blocks,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "seed": int(seed),
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "candidate_minus_reference": point_difference,
        "relative_improvement_percent": point_relative,
    }
    if len(keys) < minimum_blocks:
        return {
            "status": "insufficient_blocks",
            "reason": (
                f"at least {minimum_blocks} paired blocks are required; "
                f"received {len(keys)}"
            ),
            **base,
            "candidate_minus_reference_ci": None,
            "relative_improvement_percent_ci": None,
        }

    rng = random.Random(seed)
    differences: list[float] = []
    relative_improvements: list[float] = []
    for _ in range(n_resamples):
        draw = [keys[rng.randrange(len(keys))] for _ in keys]
        sampled_reference = fmean(reference[key] for key in draw)
        sampled_candidate = fmean(candidate[key] for key in draw)
        differences.append(sampled_candidate - sampled_reference)
        if not math.isclose(sampled_reference, 0.0, abs_tol=1e-12):
            relative_improvements.append(
                (sampled_reference - sampled_candidate)
                / abs(sampled_reference)
                * 100
            )
    differences.sort()
    relative_improvements.sort()
    tail = (1 - confidence_level) / 2
    return {
        "status": "available",
        "reason": None,
        **base,
        "candidate_minus_reference_ci": {
            "lower": _percentile(differences, tail),
            "upper": _percentile(differences, 1 - tail),
        },
        "relative_improvement_percent_ci": (
            {
                "lower": _percentile(relative_improvements, tail),
                "upper": _percentile(relative_improvements, 1 - tail),
            }
            if relative_improvements
            else None
        ),
    }


def holm_correction(
    p_values: Mapping[str, float | None],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply Holm's step-down family-wise correction deterministically."""

    if not 0 < alpha < 1:
        raise FormalAnalysisError("alpha must be between zero and one")
    available: list[tuple[str, float]] = []
    for name, raw_value in p_values.items():
        if raw_value is None:
            continue
        value = _finite(raw_value, f"p-value {name}")
        if not 0 <= value <= 1:
            raise FormalAnalysisError(f"p-value {name} must be between zero and one")
        available.append((str(name), value))
    available.sort(key=lambda item: (item[1], item[0]))
    total = len(available)
    adjusted: dict[str, dict[str, Any]] = {}
    running_adjusted = 0.0
    continue_rejecting = True
    for index, (name, raw_value) in enumerate(available):
        multiplier = total - index
        running_adjusted = max(running_adjusted, raw_value * multiplier)
        adjusted_value = min(1.0, running_adjusted)
        threshold = alpha / multiplier
        reject = continue_rejecting and raw_value <= threshold
        if not reject:
            continue_rejecting = False
        adjusted[name] = {
            "status": "available",
            "rank": index + 1,
            "raw_p_value": raw_value,
            "adjusted_p_value": adjusted_value,
            "holm_threshold": threshold,
            "reject_null": reject,
        }
    results: dict[str, dict[str, Any]] = {}
    for raw_name, raw_value in p_values.items():
        name = str(raw_name)
        results[name] = adjusted.get(
            name,
            {
                "status": "unavailable",
                "rank": None,
                "raw_p_value": None if raw_value is None else raw_value,
                "adjusted_p_value": None,
                "holm_threshold": None,
                "reject_null": None,
            },
        )
    return {
        "status": "available" if available else "insufficient_comparisons",
        "reason": None if available else "no finite p-values were supplied",
        "alpha": alpha,
        "comparison_count": total,
        "results": results,
    }


__all__ = [
    "FormalAnalysisError",
    "TemperatureObservation",
    "WarningSignal",
    "evaluate_event_warnings",
    "holm_correction",
    "paired_block_bootstrap",
]
